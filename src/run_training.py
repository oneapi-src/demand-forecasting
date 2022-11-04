# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914,E0611

"""
Run training with benchmarks.

Code adopted from
https://www.kaggle.com/code/dimitreoliveira/deep-learning-for-time-series-forecasting/notebook
"""
import argparse
import logging
import os
import pathlib
import time
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import tensorflow as tf

from utils.load_data import read_data, series_to_supervised, prepare_data
from utils.model import get_cnn_lstm

warnings.filterwarnings("ignore")
tf.compat.v1.disable_eager_execution()


def main(flags):
    """Run training with benchmarking for the CNN-LSTM model.

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

    tf.keras.utils.set_random_seed(5)

    window = 129
    lag_size = 1
    test_size = 0.3
    epochs = 10

    # create training and validation datasets
    if not os.path.exists("../data/demand/train.csv"):
        print("Data file ../data/demand/train.csv not found")
        return

    train = read_data('../data/demand/train.csv')
    series, labels = series_to_supervised(
        train,
        window=window,
        lag=lag_size
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        series,
        labels.values,
        test_size=test_size,
    )

    x_train_sub, y_train_sub = prepare_data(
        x_train.drop(['date', 'item', 'store'], axis=1), y_train, 2
    )
    x_valid_sub, y_valid_sub = prepare_data(
        x_valid.drop(['date', 'item', 'store'], axis=1), y_valid, 2
    )

    # compile the model
    model = get_cnn_lstm(x_train_sub)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=optimizer)
    model.summary()
    logger.info(
        'Starting training on %d samples with batch size %d...',
        x_train_sub.shape[0],
        flags.batch_size
    )

    # train the model using keras
    start = time.time()
    _ = model.fit(
        x=x_train_sub,
        y=y_train_sub,
        batch_size=flags.batch_size,
        validation_data=(x_valid_sub, y_valid_sub),
        epochs=epochs,
        verbose=2
    )
    end = time.time()

    train_pred = model.predict(x_train_sub, batch_size=512)
    valid_pred = model.predict(x_valid_sub, batch_size=512)
    logger.info(
        '======> Train RMSE: %.4f',
        np.sqrt(mean_squared_error(y_train_sub, train_pred))
    )
    logger.info(
        '======> Validation RMSE: %.4f',
        np.sqrt(mean_squared_error(y_valid_sub, valid_pred))
    )

    logger.info(
        '=======> Train time : %d seconds',
        end - start
    )

    if flags.save_model_dir is not None:
        path = pathlib.Path(flags.save_model_dir)
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model...")
        model.save(flags.save_model_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")

    parser.add_argument('-s',
                        '--save_model_dir',
                        default=None,
                        type=str,
                        required=False,
                        help="directory to save model to"
                        )

    parser.add_argument('-b',
                        '--batch_size',
                        default=512,
                        required=False,
                        type=int,
                        help="training batch size"
                        )

    FLAGS = parser.parse_args()
    main(FLAGS)
