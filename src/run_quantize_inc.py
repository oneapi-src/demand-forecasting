# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914,C0115,C0116


"""
Quantize a model using Intel Neural Compressor
"""

import argparse
import os
import pathlib

from neural_compressor.experimental import Quantization, common
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils.load_data import read_data, series_to_supervised, prepare_data

WINDOW = 129
LAG_SIZE = 1
BATCH_SIZE = 10000
tf.keras.utils.set_random_seed(5)


class Dataset:

    def __init__(self):

        train = read_data('../data/demand/train.csv')

        series, labels = series_to_supervised(
            train,
            window=WINDOW,
            lag=LAG_SIZE
        )

        _, x_valid, _, y_valid = train_test_split(
            series,
            labels.values,
            test_size=0.3,
            random_state=0
        )
        x_valid_sub, y_valid_sub = prepare_data(
            x_valid.drop(['date', 'item', 'store'], axis=1), y_valid, 2
        )

        self.x_valid = x_valid_sub
        self.y_valid = y_valid_sub

    def __getitem__(self, index):
        return self.x_valid[index], self.y_valid[index]

    def __len__(self):
        return len(self.x_valid)


class RMSEMetric:

    def __init__(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def update(self, predict, label):
        self.pred_list.extend(predict.reshape(-1, 1))
        self.label_list.extend(label)
        self.samples += len(label)

    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.samples = 0

    def result(self):
        return np.sqrt(mean_squared_error(self.pred_list, self.label_list))


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


def quantize_model(input_graph_path, dataset, inc_config_file):
    """Quantizes the model using the given dataset and INC config

    Args:
        input_graph_path: path to .pb model.
        dataset : Dataset to use for quantization.
        inc_config_file : Path to INC config.
    """
    quantizer = Quantization(inc_config_file)
    quantizer.calib_dataloader = common.DataLoader(
        dataset, batch_size=BATCH_SIZE
    )
    quantizer.eval_dataloader = common.DataLoader(
        dataset, batch_size=BATCH_SIZE
    )
    quantizer.metric = common.Metric(RMSEMetric)
    quantizer.model = input_graph_path
    quantized_model = quantizer.fit()

    return quantized_model


def main(flags) -> None:
    """Calibrate model for int 8 and serialize as a .pt

    Args:
        flags: benchmarking flags
    """

    if not os.path.exists(flags.saved_frozen_graph):
        print("Saved model %s not found!", flags.saved_frozen_graph)
        return

    if not os.path.exists(flags.inc_config_file):
        print("INC configuration %s not found!", flags.inc_config_file)
        return

    dataset = Dataset()
    quantized_model = quantize_model(
        flags.saved_frozen_graph, dataset, flags.inc_config_file)

    path = pathlib.Path(flags.output_dir)
    path.mkdir(parents=True, exist_ok=True)

    quantized_model.save(
        os.path.join(flags.output_dir, "saved_frozen_int8_model.pb")
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--saved_frozen_graph',
        required=True,
        help="saved pretrained frozen graph to quantize",
        type=str
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        help="directory to save quantized model.",
        type=str
    )

    parser.add_argument(
        '--inc_config_file',
        help="INC conf yaml",
        required=True
    )

    FLAGS = parser.parse_args()

    main(FLAGS)
