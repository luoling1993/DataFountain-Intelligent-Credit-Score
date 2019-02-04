#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_absolute_error

h2o.init(max_mem_size_GB=12)

BASE_PATH = os.path.join('data')
RAWDATA_PATH = os.path.join(BASE_PATH, 'RawData')
ETLDATA_PATH = os.path.join(BASE_PATH, 'EtlData')


def get_data():
    data_name = 'features.csv'
    data_name = os.path.join(ETLDATA_PATH, data_name)
    df = h2o.upload_file(data_name)

    train_data = df[df['score'] > 0]
    test_data = df[df['score'] < 0]

    return train_data, test_data


def auto_ml():
    train_data, test_data = get_data()

    remove_columns = ['id', 'score']
    features_columns = [column for column in train_data.columns if column not in remove_columns]

    test_sample_ids = test_data['id'].as_data_frame().values.flatten()
    y_labels = train_data['score'].as_data_frame().values.flatten()

    aml = H2OAutoML(max_models=50, seed=2019, max_runtime_secs=7200)
    aml.train(x=features_columns, y='score', training_frame=train_data)

    lb = aml.leaderboard
    print(lb.head())

    train_data = train_data[features_columns]
    y_train = aml.predict(train_data).as_data_frame().values.flatten()
    mae_error = mean_absolute_error(y_labels, y_train)
    print(f'train mae error: {mae_error}')
    fold_score = 1 / (1 + mae_error)
    print(f'fold score is {fold_score}')

    test_data = test_data[features_columns]
    automl_predictions = aml.predict(test_data).as_data_frame().values.flatten()

    df = pd.DataFrame({'id': test_sample_ids,
                       'score': automl_predictions})

    df['score'] = df['score'].apply(lambda item: round(item))
    df['score'] = df['score'].apply(lambda item: int(item))
    df.to_csv('submittion_automl.csv', index=False, header=True)


if __name__ == '__main__':
    auto_ml()
