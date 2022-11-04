# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914,E0611,W0108

"""
Convert a keras saved model to a frozen graph.
"""
import argparse
import os
import pathlib
import warnings

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2

warnings.filterwarnings("ignore")


def main(flags):
    """Convert a keras saved model to a frozen graph.

    Args:
        flags : run flags
    """
    model = tf.keras.models.load_model(flags.keras_saved_model_dir)

    full_model = tf.function(lambda x: model(x))
    concrete_function = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    path = pathlib.Path(flags.output_saved_dir)
    path.mkdir(parents=True, exist_ok=True)

    tf.io.write_graph(
        graph_or_graph_def=frozen_model.graph,
        logdir='.',
        name=os.path.join(flags.output_saved_dir, "saved_frozen_model.pb"),
        as_text=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        '--keras_saved_model_dir',
                        default=None,
                        type=str,
                        required=True,
                        help="directory with saved keras model."
                        )

    parser.add_argument('-o',
                        '--output_saved_dir',
                        default=None,
                        type=str,
                        required=True,
                        help="directory to save frozen graph to."
                        )

    FLAGS = parser.parse_args()
    main(FLAGS)
