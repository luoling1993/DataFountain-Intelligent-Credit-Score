#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings(action='ignore')

BASE_PATH = os.path.join('data')
RAWDATA_PATH = os.path.join(BASE_PATH, 'RawData')
ETLDATA_PATH = os.path.join(BASE_PATH, 'EtlData')


class LGBOptimize(object):
    @staticmethod
    def _get_data():
        data_name = os.path.join(ETLDATA_PATH, 'features.csv')
        df = pd.read_csv(data_name, header=0)
        df = df[df['score'] > 0]
        return df

    def optimize(self):
        dataset = self._get_data()

        remove_columns = ['id', 'score']
        x_columns = [column for column in dataset.columns if column not in remove_columns]

        x_data = dataset[x_columns]
        y_data = dataset['score']

        """n_estimators: best:239 best_score: 14.847823004441079"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 50,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # dtrain = lgb.Dataset(x_data, y_data)
        # cv_results = lgb.cv(params, dtrain, num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
        #                     metrics='mae', early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=2018)
        # print('best n_estimators:', len(cv_results['l1-mean']))
        # print('best cv score:', cv_results['l1-mean'][-1])

        """max_depth:6 num_leaves:31 best_score:14.803535507027162"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'n_estimators': 239,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 50,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # grid_params = {
        #     'max_depth': [6],
        #     'num_leaves': [28, 29, 30, 31, 32, 33, 34, 35]
        # }
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                            n_jobs=5)
        # grid_search.fit(x_data, y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        """min_child_samples:43 min_child_weight:0 best_score:14.783911433202508"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'n_estimators': 239,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 31,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # grid_params = {
        #     'min_child_samples': [43],
        #     'min_child_weight': [0, 0.001]
        # }
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                            n_jobs=5)
        # grid_search.fit(x_data, y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        """subsample:0.32 colsample_bytree:0.36 best_score:14.771928920921576"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'n_estimators': 239,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'min_child_samples': 43,
        #     'min_child_weight': 0,
        #     'num_leaves': 65,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # grid_params = {
        #     'subsample': [0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5],
        #     'colsample_bytree': [0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]
        # }
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                            n_jobs=5)
        # grid_search.fit(x_data, y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        """reg_alpha:2 reg_lambda:0.1 best_score:14.75515862949816"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'n_estimators': 239,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'min_child_samples': 43,
        #     'min_child_weight': 0,
        #     'num_leaves': 65,
        #     'max_depth': 6,
        #     'subsample': 0.32,
        #     'colsample_bytree': 0.36,
        # }
        # grid_params = {
        #     'reg_alpha': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3],
        #     'reg_lambda': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 3],
        # }
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                            n_jobs=5)
        # grid_search.fit(x_data, y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        """learning_rate:0.1 best_score:14.778696016248404"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'mae',
        #     'n_estimators': 239,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'min_child_samples': 43,
        #     'min_child_weight': 0,
        #     'num_leaves': 65,
        #     'max_depth': 6,
        #     'subsample': 0.32,
        #     'colsample_bytree': 0.36,
        #     'reg_alpha': 2,
        #     'reg_lambda': 0.1,
        # }
        # grid_params = {
        #     'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
        # }
        # gbm = lgb.LGBMRegressor(**params)
        # grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                            n_jobs=5)
        # grid_search.fit(x_data, y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        params = {
            'boosting_type': 'gbdt',
            'objective': 'mae',
            'n_estimators': 10000,
            'metric': 'mae',
            'learning_rate': 0.01,
            'min_child_samples': 46,
            'min_child_weight': 0.01,
            'subsample_freq': 1,
            'num_leaves': 40,
            'max_depth': 7,
            'subsample': 0.42,
            'colsample_bytree': 0.48,
            'reg_alpha': 2,
            'reg_lambda': 0.1,
            'verbose': -1,
            'seed': 4590
        }
        grid_params = {
            'subsample': [0.45, 0.5, 0.55],
            'colsample_bytree': [0.85, 0.9, 0.95]
        }
        gbm = lgb.LGBMRegressor(**params)
        grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
                                   n_jobs=5)
        grid_search.fit(x_data, y_data)
        print(f'best params: {grid_search.best_params_}')
        print(f'best score: {grid_search.best_score_}')


class XGBOptimize(object):
    @staticmethod
    def _get_data():
        data_name = os.path.join(ETLDATA_PATH, 'features.csv')
        df = pd.read_csv(data_name, header=0)
        df = df[df['score'] > 0]
        return df

    def optimize(self):
        dataset = self._get_data()

        remove_columns = ['id', 'score']
        x_columns = [column for column in dataset.columns if column not in remove_columns]

        x_data = dataset[x_columns]
        y_data = dataset['score']

        """n_estimators: best:154 best_score: 14.819362"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'reg:linear',
        #     'learning_rate': 0.1,
        #     'num_leaves': 50,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        # dtrain = xgb.DMatrix(x_data, y_data)
        # cv_results = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
        #                     metrics='mae', early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=2018)
        # print('best n_estimators:', len(cv_results['test-mae-mean']))
        # print('best cv score:', cv_results['test-mae-mean'][-1])

        """max_depth:5 num_leaves:40 best_score:14.869131424560546"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'reg:linear',
        #     'n_estimators': 154,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 50,
        #     'max_depth': 6,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        #
        # grid_params = {
        #     'max_depth': [5],
        #     'num_leaves': [40]
        # }
        # gbm = xgb.XGBRegressor(**params)
        # grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                            n_jobs=5)
        # grid_search.fit(x_data, y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        """min_child_samples:46 min_child_weight:0.01 best_score:14.81476454512563"""
        # params = {
        #     'boosting_type': 'gbdt',
        #     'objective': 'reg:linear',
        #     'n_estimators': 154,
        #     'metric': 'mae',
        #     'learning_rate': 0.1,
        #     'num_leaves': 40,
        #     'max_depth': 5,
        #     'subsample': 0.8,
        #     'colsample_bytree': 0.8,
        # }
        #
        # grid_params = {
        #
        # }
        # gbm = xgb.XGBRegressor(**params)
        # grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
        #                            n_jobs=5)
        # grid_search.fit(x_data, y_data)
        # print(f'best params: {grid_search.best_params_}')
        # print(f'best score: {grid_search.best_score_}')

        """max_depth: min_child_weight"""
        params = {
            'boosting_type': 'gbdt',
            'objective': 'reg:linear',
            'n_estimators': 154,
            'learning_rate': 0.1,
            'num_leaves': 40,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

        grid_params = {
            'max_depth': range(4, 5, 6),
            'min_child_weight': [0, 1, 2]
        }
        gbm = xgb.XGBRegressor(**params)
        grid_search = GridSearchCV(gbm, param_grid=grid_params, scoring='neg_mean_absolute_error', cv=5, verbose=1,
                                   n_jobs=5)
        grid_search.fit(x_data, y_data)
        print(f'best params: {grid_search.best_params_}')
        print(f'best score: {grid_search.best_score_}')


if __name__ == '__main__':
    optimizer = LGBOptimize()
    optimizer.optimize()
