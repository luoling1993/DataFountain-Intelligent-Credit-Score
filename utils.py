#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

BASE_PATH = os.path.join('data')
RAWDATA_PATH = os.path.join(BASE_PATH, 'RawData')
ETLDATA_PATH = os.path.join(BASE_PATH, 'EtlData')


def get_data():
    data_name = os.path.join(ETLDATA_PATH, 'features.csv')
    df = pd.read_csv(data_name, header=0)
    return df


def get_ensemble_score(score_df):
    score_df = score_df.copy()
    model_list = list()
    for column in score_df.columns:
        if column in ['id', 'score']:
            continue

        model_list.append(column)
        model_names = ' + '.join(model_list)

        score_df['helper'] = np.mean(score_df[model_list], axis=1)
        mae = mean_absolute_error(score_df['score'], score_df['helper'])
        mae_score = 1 / (1 + mae)

        print(f'model name: {model_names}, mae: {mae}, score: {mae_score}')


def get_combinations(arr_list):
    combinations_list = list()
    length = len(arr_list)
    for num in range(2, length):
        for bin_item in combinations(arr_list, num):
            combinations_list.append(list(bin_item))

    return combinations_list


def get_values_by_index(value_list, index_list):
    new_values = list()

    for index, value in enumerate(value_list):
        if index in index_list:
            new_values.append(value)
    return new_values


def get_score_array(dataset):
    dataset = dataset.copy()
    score_array = list()

    remove_columns = ['id', 'score']
    for column in dataset.columns:
        if column in remove_columns:
            continue
        score_array.append(np.array(dataset[column]))

    return score_array


def get_blending_score(score_array, weights):
    weight_prediction = 0.0
    for weight, prediction in zip(weights, score_array):
        weight_prediction += weight * prediction
    return weight_prediction
