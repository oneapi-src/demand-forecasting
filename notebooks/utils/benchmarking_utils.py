# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
"""
Process raw dataset for experiments
"""

import os
from tqdm import tqdm
from contextlib import redirect_stdout
import sys
sys.path.append("../../src")
from src import run_training
from src import run_inference
import argparse
from importlib import reload
import logging
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from subprocess import check_output, STDOUT


def run_training_benchmark(intel=False, iterations = 1):
    FLAGS = argparse.Namespace()
    if intel:
        subfolder_name = 'intel'
    else:
        subfolder_name = 'stock'
        
    test_params = [
        ["64",64],
        ["128",128],
        ["256",256]
    ]

    for k in tqdm(range(1,iterations+1), desc='iteration'):
        for foldername, initialsize in tqdm(test_params, desc='experiment'):
            os.makedirs(f"../logs/{foldername}/{subfolder_name}", exist_ok=True)

            FLAGS.logfile = f"../logs/{foldername}/{subfolder_name}/performance.log"
            fairness_logfile = f"../logs/{foldername}/{subfolder_name}/fairness.log"

            for j in tqdm(range(1,5), desc='train'):
                reload(logging)
                FLAGS.batch_size = initialsize
                FLAGS.save_model_dir= f"../saved_models/{foldername}/{subfolder_name}"
                with open(fairness_logfile, 'a') as f:
                    with redirect_stdout(f):\
                        run_training.main(FLAGS)


def convert_keras_to_frozen_graph_benchmark(intel=False, iterations = 1):
    FLAGS = argparse.Namespace()
    if intel:
        subfolder_name = 'intel'
    else:
        subfolder_name = 'stock'

    test_params = [
        "64",
        "128",
        "256",
    ]

    for k in tqdm(range(1,iterations+1), desc='iteration'):
        for foldername in tqdm(test_params, desc='experiment'):
            # Convert the trained model to a frozen graph
            keras_saved_model_dir = f"../saved_models/{foldername}/{subfolder_name}/"
            output_saved_dir = f"../saved_models/{foldername}/{subfolder_name}/"      
            command = f"python convert_keras_to_frozen_graph.py -s {keras_saved_model_dir} -o {output_saved_dir}"
            check_output(command, shell=True, stderr=STDOUT)


def run_inference_benchmark(intel=False, iterations = 1):
    FLAGS = argparse.Namespace()
    FLAGS.input_file = "../data/demand/test_full.csv"
    test_params = [
        ["64",64],
        ["128",128],
        ["256",256]
    ]

    if intel:
        subfolder_name = 'intel'
    else:
        subfolder_name = 'stock'

    for k in tqdm(range(1,iterations+1), desc='iteration'):
        for foldername, testsize in tqdm(test_params, desc='experiment'):
            
            FLAGS.logfile = f"../logs/{foldername}/{subfolder_name}/inference.log"
            fairness_logfile = f"../logs/{foldername}/{subfolder_name}/inference_fairness.log"

            for j in tqdm(range(1,5), desc='inference'):
                reload(logging)
                FLAGS.batch_size  = testsize
                FLAGS.benchmark_mode  = True
                FLAGS.num_iters  = 1000
                FLAGS.saved_frozen_model = f"../saved_models/{foldername}/{subfolder_name}/saved_frozen_model.pb"
                with open(fairness_logfile, 'a') as f:
                    with redirect_stdout(f):\
                        run_inference.main(FLAGS)


def run_inference_quantized_model_benchmark(intel=True, iterations = 1):
    FLAGS = argparse.Namespace()
    FLAGS.input_file = "../data/demand/test_full.csv"
    test_params = [
        ["64",64],
        ["128",128],
        ["256",256]
    ]

    if intel:
        subfolder_name = 'intel'
    else:
        subfolder_name = 'stock'

    for k in tqdm(range(1,iterations+1), desc='iteration'):
        for foldername, testsize in tqdm(test_params, desc='experiment'):
            
            FLAGS.logfile = f"../logs/{foldername}/{subfolder_name}/inc_inference.log"
            fairness_logfile = f"../logs/{foldername}/{subfolder_name}/inc_inference_fairness.log"

            for j in tqdm(range(1,5), desc='inference'):
                reload(logging)
                FLAGS.batch_size  = testsize
                FLAGS.benchmark_mode  = True
                FLAGS.num_iters  = 1000
                FLAGS.saved_frozen_model = f"../saved_models/{foldername}/{subfolder_name}/saved_frozen_int8_model.pb"
                with open(fairness_logfile, 'a') as f:
                    with redirect_stdout(f):\
                        run_inference.main(FLAGS)

def run_quantize_inc_benchmark(iterations = 1):
    FLAGS = argparse.Namespace()
    FLAGS.input_file = "../data/demand/test_full.csv"
    test_params = [
        "64",
        "128",
        "256"
    ]

    subfolder_name = 'intel'

    for k in tqdm(range(1,iterations+1), desc='iteration'):
        for foldername in tqdm(test_params, desc='experiment'):
            # Quantize the trained model
            saved_frozen_graph = f"../saved_models/{foldername}/{subfolder_name}/saved_frozen_model.pb"
            output_dir = f"../saved_models/{foldername}/{subfolder_name}"
            inc_config_file = 'conf.yaml'    
            command = f"python run_quantize_inc.py --saved_frozen_graph {saved_frozen_graph} --output_dir {output_dir} --inc {inc_config_file} "
            check_output(command, shell=True, stderr=STDOUT)


def load_results_dict_training():
    results_dict = defaultdict(dict)
    subfolder_names = {'stock':'stock','intel':'intel'}
    foldernames = {'64':'64','128':'128','256':'256'}
    for experiment_n, foldername in enumerate(foldernames.keys()):
        for increment_n in range(1,2):
            results_dict['Experiment'][experiment_n * 3 + increment_n] = experiment_n + 1
            for subfolder_name in subfolder_names.keys():
                logfile = f"../logs/{foldername}/{subfolder_name}/performance.log"
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                filtered_lines = [line for line in lines if line.find('time') != -1]

                results_dict['Batch Size'][experiment_n * 3 + increment_n] = foldernames[foldername]
                results_dict['Round'][experiment_n * 3 + increment_n] = f'{increment_n}'
                time = np.mean([float(re.findall("\d+",filtered_lines[i])[0]) for i in range(increment_n-1,4)])
                results_dict[subfolder_names[subfolder_name]][experiment_n * 3 + increment_n] = time
            stock_time = results_dict[subfolder_names['stock']][experiment_n * 3 + increment_n]
            stock_intel = results_dict[subfolder_names['intel']][experiment_n * 3 + increment_n]
            results_dict['Intel speedup over stock'][experiment_n * 3 + increment_n] = stock_time / stock_intel
    return results_dict

def load_results_dict_inference():
    results_dict = defaultdict(dict)
    subfolder_names = {'stock':'stock','intel':'intel'}
    foldernames = {'64':'64','128':'128','256':'256'}
    for experiment_n, foldername in enumerate(foldernames.keys()):
        for increment_n in range(1,2):
            results_dict['Experiment'][experiment_n * 3 + increment_n] = experiment_n + 1
            for subfolder_name in subfolder_names.keys():
                logfile = f"../logs/{foldername}/{subfolder_name}/inference.log"
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                    
                results_dict['Batch Size'][experiment_n * 3 + increment_n] = foldernames[foldername]
                results_dict['Round'][experiment_n * 3 + increment_n] = f'{increment_n}'
                
                start = (increment_n - 1) * 8
                end = start + 8 

                if subfolder_name == 'stock':
                    filtered_lines = lines
                    time = np.mean([float(re.findall("\d+.\d+",filtered_lines[i])[0]) for i in range(start+1,end,2)])
                    results_dict[subfolder_names[subfolder_name]][experiment_n * 3 + increment_n] = time
                else:
                    filtered_lines = lines
                    inc_logfile = f"../logs/{foldername}/{subfolder_name}/inc_inference.log"
                    with open(inc_logfile, 'r') as f:
                        inc_lines = f.readlines()
                    filtered_lines_daal = inc_lines
                    time = np.mean([float(re.findall("\d+.\d+",filtered_lines[i])[0]) for i in range(start+1,end,2)])
                    results_dict[subfolder_names[subfolder_name]][experiment_n * 3 + increment_n] = time
                    
                    time = np.mean([float(re.findall("\d+.\d+",filtered_lines_daal[i])[0]) for i in range(start+1,end,2)])
                    results_dict['inc'][experiment_n * 3 + increment_n] = float(time)
                            
            results_dict['Intel speedup over stock'][experiment_n * 3 + increment_n] = results_dict[subfolder_names['stock']][experiment_n * 3 + increment_n] / results_dict[subfolder_names['intel']][experiment_n * 3 + increment_n]
            results_dict['Intel speedup over stock:inc'][experiment_n * 3 + increment_n] = results_dict[subfolder_names['stock']][experiment_n * 3 + increment_n] / results_dict['inc'][experiment_n * 3 + increment_n]
    return results_dict

def print_inference_benchmark_table():
    df = pd.DataFrame(load_results_dict_inference())
    df = df.round(4)
    df['stock'] = df['stock'].apply(lambda x:str(x)+'s')
    df['intel'] = df['intel'].apply(lambda x:str(x)+'s')
    df['inc'] = df['inc'].apply(lambda x:str(x)+'s')
    df['% gain:intel'] = df['Intel speedup over stock'].apply(lambda x:str(round(x-1,2))+'%')
    df['% gain:inc'] = df['Intel speedup over stock:inc'].apply(lambda x:str(round(x-1,2))+'%')
    df['Intel speedup over stock'] = df['Intel speedup over stock'].apply(lambda x:str(x)+'x')
    df['Intel speedup over stock:inc'] = df['Intel speedup over stock:inc'].apply(lambda x:str(x)+'x')
    return df

def print_training_benchmark_table():
    df = pd.DataFrame(load_results_dict_training())
    df = df.round(2)
    df['stock'] = df['stock'].apply(lambda x:str(x)+'s')
    df['intel'] = df['intel'].apply(lambda x:str(x)+'s')
    df['% gain'] = df['Intel speedup over stock'].apply(lambda x:str(round(x-1,2))+'&')
    df['Intel speedup over stock'] = df['Intel speedup over stock'].apply(lambda x:str(x)+'x')
    return df

def print_training_benchmark_bargraph():
    df = pd.DataFrame(load_results_dict_training())
    fig, (ax1) = plt.subplots(1,1,figsize=[14,6])
    fig.suptitle('Incremental Training Performance Gain Relative to Stock')
    x = np.arange(3)  # the label locations
    width = 0.35 
    size_list = ['64','128','256']
    ax1.set_ylabel('Relative Performance to Stock \n (Higher is better)')
    for i,ax in enumerate([ax1]):
        curslice = slice(i*3,(i+1)*3)
        xbg081 = round(df['stock'].iloc()[curslice]/df['stock'].iloc()[curslice],2)
        xbg151 = round(df['stock'].iloc()[curslice]/df['intel'].iloc()[curslice],2)
        rects1 = ax.bar(x - width/2, xbg081, width, label='stock', color='b')
        rects2 = ax.bar(x + width/2, xbg151, width, label='intel', color='deepskyblue')
        ax.bar_label(rects1, labels=[str(i) + 'x' for i in xbg081], padding=3)
        ax.bar_label(rects2, labels=[str(i) + 'x' for i in xbg151], padding=3)
        ax.set_xticks(x, [f'Experiment 1\n Batch Size = {size_list[0]}',\
            f'Experiment 2\n Batch Size = {size_list[1]}',\
            f'Experiment 3\n Batch Size = {size_list[2]}'])
        ax.set_ylim([0, 2])
    ax1.legend()

def print_inference_benchmark_bargraph():
    df = pd.DataFrame(load_results_dict_inference())
    fig, (ax1) = plt.subplots(1,1,figsize=[14,6])
    fig.suptitle('Inference Performance Gain Relative to Stock')
    x = np.arange(3)  # the label locations
    width = 0.25 
    size_list = ['64','128','256']
    ax1.set_ylabel('Inference Performance Gain Relative to Stock \n (Higher is better)')
    for i,ax in enumerate([ax1]):
        curslice = slice(i*3,(i+1)*3)
        xbg081 = round(df['stock'].iloc()[curslice]/df['stock'].iloc()[curslice],2)
        xbg151 = round(df['stock'].iloc()[curslice]/df['intel'].iloc()[curslice],2)
        daal = round(df['stock'].iloc()[curslice]/df['inc'].iloc()[curslice],2)
        rects1 = ax.bar(x - width/2, xbg081, width, label='stock', color='b')
        rects2 = ax.bar(x + width/2, xbg151, width, label='intel', color='deepskyblue')
        rects3 = ax.bar(x + width/2*3, daal, width, label='inc', color='orange')
        ax.bar_label(rects1, labels=[str(i) + 'x' for i in xbg081], padding=3)
        ax.bar_label(rects2, labels=[str(i) + 'x' for i in xbg151], padding=3)
        ax.bar_label(rects3, labels=[str(i) + 'x' for i in daal], padding=3)
        ax.set_xticks(x, [f'Experiment 1\nBatch Inference\n(N = {size_list[0]})',\
            f'Experiment 2\nBatch Inference\n(N = {size_list[1]})',\
            f'Experiment 3\nBatch Inference\n(N = {size_list[2]})'])
        ax.set_ylim([0, 3])
    ax1.legend()