# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914,E1129,E0611

"""
Run inference with benchmarks on Tensorflow native models.

Code adopted from
https://www.kaggle.com/code/dimitreoliveira/deep-learning-for-time-series-forecasting/notebook
"""
import argparse
import logging
import math
import os
import pathlib
import time
import warnings

import numpy as np
import tensorflow as tf

from utils.load_data import read_data, series_to_supervised, prepare_data

warnings.filterwarnings("ignore")
tf.keras.utils.set_random_seed(42)


def load_pb(in_model: str) -> tf.compat.v1.Graph:
    """Load a frozen graph from a .pb file

    Args:
        in_model (str): .pb file

    Returns:
        tf.compat.v1.Graph: tensorflow graph version
    """
    detection_graph = tf.compat.v1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(in_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(od_graph_def, name='')
    return detection_graph


def get_concrete_function(graph_def: tf.compat.v1.Graph):
    """Get a concrete function from a TF graph to
    make a callable

    Args:
        graph_def (tf.compat.v1.Graph): Graph to turn into a callable
    """

    def imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrap_function = tf.compat.v1.wrap_function(imports_graph_def, [])
    graph = wrap_function.graph

    return wrap_function.prune(
        tf.nest.map_structure(graph.as_graph_element, ["x:0"]),
        tf.nest.map_structure(graph.as_graph_element, ["Identity:0"]))


def main(flags):
    """Run inference with benchmarking for the CNN-LSTM model.

    Args:
        flags : run flags
    """

    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(flags.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
    logger = logging.getLogger()

    # prepare data for prediction
    window = 129
    lag_size = 1
    batch_size = flags.batch_size
    num_iters = flags.num_iters
    input_file = flags.input_file

    # read in data files
    if not os.path.exists(input_file):
        print(f"Data file {input_file} not found")
        return

    test = read_data(input_file)
    series, labels = series_to_supervised(
        test,
        window=window,
        lag=lag_size
    )

    # load model which is saved as a frozen graph
    model = load_pb(flags.saved_frozen_model)
    concrete_function = get_concrete_function(
        graph_def=model.as_graph_def()
    )

    if flags.benchmark_mode:
        series = series[labels == -1]
        labels = labels[labels == -1]

        x_test, x_labels = series, labels.values
        x_test_sub, _ = prepare_data(
            x_test.drop(['date', 'item', 'store'], axis=1), x_labels, 2
        )
        # benchmark execution runtime
        logger.info(
            'Starting inference on batch size %d for %d iterations',
            batch_size,
            num_iters
        )
        times = []
        for i in range(10+num_iters):
            idx = np.random.randint(x_test_sub.shape[0], size=batch_size)
            btch = tf.constant(x_test_sub[idx], dtype=tf.float32)
            start = time.time()
            res = concrete_function(x=btch)
            end = time.time()
            if i > 10:
                times.append(end - start)
        logger.info(
            '=======> Average Inference Time : %f seconds',
            np.mean(times)
        )
    else:
        series = series[(labels == -1) & (series['sales(t)'] != -1)]
        labels = labels[labels == -1]

        x_test, x_labels = series, labels.values
        x_test_sub, _ = prepare_data(
            x_test.drop(['date', 'item', 'store'], axis=1), x_labels, 2
        )
        predictions = []
        for i in range(math.ceil(len(x_test_sub)/batch_size)):
            print(x_test_sub.shape)
            btch = tf.constant(
                x_test_sub[batch_size * i: batch_size * (i+1)],
                dtype=tf.float32)
            res = concrete_function(x=btch)[0]
            predictions.append(res.numpy())
        out_df = x_test[['date', 'item', 'store']]
        out_df['prediction'] = np.vstack(predictions)
        print(out_df.to_json(orient="records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-s',
                        '--saved_frozen_model',
                        default=None,
                        type=str,
                        required=False,
                        help="saved frozen graph."
                        )

    parser.add_argument('-b',
                        '--batch_size',
                        default=1,
                        type=int,
                        required=False,
                        help="batch size to use"
                        )

    parser.add_argument(
        '--input_file',
        type=str,
        required=True
    )

    parser.add_argument(
        '--benchmark_mode',
        action="store_true",
        default=False,
        help="benchmark inference time"
    )

    parser.add_argument('-n',
                        '--num_iters',
                        default=100,
                        type=int,
                        required=False,
                        help="number of iterations to use when benchmarking"
                        )

    FLAGS = parser.parse_args()
    main(FLAGS)
