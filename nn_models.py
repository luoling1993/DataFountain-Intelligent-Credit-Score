#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from keras.layers import Dense, ReLU, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error

from utils import get_data


class NNModels(object):
    def __init__(self):
        pass

    @staticmethod
    def _get_nn_base_model(input_dim=831):
        nn_model = Sequential()

        nn_model.add(Dense(400, input_dim=input_dim, kernel_initializer='normal'))
        nn_model.add(ReLU())
        nn_model.add(BatchNormalization())
        nn_model.add(Dropout(0.5))

        nn_model.add(Dense(200, kernel_initializer='normal'))
        nn_model.add(ReLU())
        nn_model.add(BatchNormalization())
        nn_model.add(Dropout(0.5))

        nn_model.add(Dense(50, kernel_initializer='normal'))
        nn_model.add(ReLU())
        nn_model.add(BatchNormalization())
        nn_model.add(Dropout(0.5))

        nn_model.add(Dense(1, kernel_initializer='normal'))
        nn_model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        return nn_model

    def nn_model(self):
        dataset = get_data()

        train_data = dataset[dataset['score'] > 0.0]
        test_data = dataset[dataset['score'] < 0]
        y_data = train_data['score']
        x_data = train_data.drop(columns=['id', 'score'])

        test_data.reset_index(inplace=True, drop=True)
        x_test = test_data.drop(columns=['id', 'score'])

        baseline_model = self._get_nn_base_model
        estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)

        # kfold = KFold(n_splits=5)
        # mae = make_scorer(mean_absolute_error)
        # res = cross_val_score(estimator, X=x_data, y=y_data, cv=kfold, scoring=mae)
        # mae_error = np.mean(res)
        estimator.fit(x_data, y_data)
        y_pred = estimator.predict(x_data)
        mae_error = mean_absolute_error(y_pred, y_data)

        print(f'mae error: {mae_error}')
        print(f'nn score: {1 / (1 + mae_error)}')

        pred = estimator.predict(x_test)
        sub = pd.DataFrame({'id': test_data['id'], 'score': pred})
        sub['score'] = sub['score'].apply(lambda item: int(round(item)))
        sub.to_csv('submittion_5.csv', index=False)


NNModels().nn_model()
