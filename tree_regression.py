#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import inspect
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import sparse
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from utils import get_data
from utils import timer

warnings.filterwarnings(action='ignore')


class TreeRegression(object):
    def __init__(self, mode, n_fold=10, seed=4590, save=False):
        self.mode = mode
        self.n_fold = n_fold
        self.seed = seed
        self.save = save
        self._check_mode(self.mode)

    @staticmethod
    def _check_mode(mode):
        assert mode in ['lgb', 'xgb', 'rf', 'ctb', 'ada', 'gbdt']

    def _get_gbm(self, params):
        if self.mode == 'lgb':
            gbm = LGBMRegressor(**params)
        elif self.mode == 'xgb':
            gbm = XGBRegressor(**params)
        elif self.mode == 'ctb':
            gbm = CatBoostRegressor(**params)
        elif self.mode == 'ada':
            gbm = AdaBoostRegressor(**params)
        elif self.mode == 'gbdt':
            gbm = GradientBoostingRegressor(**params)
        elif self.mode == 'rf':
            gbm = RandomForestRegressor(**params)
        else:
            raise ValueError()
        return gbm

    @staticmethod
    def _get_dataset():
        dataset = get_data()

        train_data = dataset[dataset['score'] > 0.0]
        test_data = dataset[dataset['score'] < 0.0]

        train_data.reset_index(inplace=True, drop=True)
        test_data.reset_index(inplace=True, drop=True)

        return train_data, test_data

    @staticmethod
    def _get_iteration_kwargs(gbm):
        predict_args = inspect.getfullargspec(gbm.predict).args
        if hasattr(gbm, 'best_iteration_'):
            best_iteration = getattr(gbm, 'best_iteration_')
            if 'num_iteration' in predict_args:
                iteration_kwargs = {'num_iteration': best_iteration}
            elif 'ntree_end' in predict_args:
                iteration_kwargs = {'ntree_end': best_iteration}
            else:
                raise ValueError()
        elif hasattr(gbm, 'best_ntree_limit'):
            best_iteration = getattr(gbm, 'best_ntree_limit')
            if 'ntree_limit' in predict_args:
                iteration_kwargs = {'ntree_limit': best_iteration}
            else:
                raise ValueError()
        else:
            raise ValueError()
        return iteration_kwargs

    def _ensemble_tree(self, params):
        train_data, test_data = self._get_dataset()

        columns = train_data.columns

        remove_columns = ['id', 'score']
        features_columns = [column for column in columns if column not in remove_columns]

        train_labels = train_data['score']
        train_x = train_data[features_columns]
        test_x = test_data[features_columns]

        # to csr 加快模型速度
        train_x = sparse.csr_matrix(train_x.values)
        test_x = sparse.csr_matrix(test_x.values)

        kfolder = KFold(n_splits=self.n_fold, shuffle=True, random_state=self.seed)
        kfold = kfolder.split(train_x, train_labels)

        preds_list = list()
        oof = np.zeros(train_data.shape[0])
        for train_index, vali_index in kfold:
            k_x_train = train_x[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_vali = train_x[vali_index]
            k_y_vali = train_labels.loc[vali_index]

            gbm = self._get_gbm(params)
            gbm = gbm.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                          early_stopping_rounds=200, verbose=False)
            iteration_kwargs = self._get_iteration_kwargs(gbm)
            k_pred = gbm.predict(k_x_vali, **iteration_kwargs)
            oof[vali_index] = k_pred

            preds = gbm.predict(test_x, **iteration_kwargs)
            preds_list.append(preds)

        fold_mae_error = mean_absolute_error(train_labels, oof)
        print(f'{self.mode} fold mae error is {fold_mae_error}')
        fold_score = 1 / (1 + fold_mae_error)
        print(f'fold score is {fold_score}')

        preds_columns = ['preds_{id}'.format(id=i) for i in range(self.n_fold)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        preds_list = list(preds_df.mean(axis=1))
        prediction = preds_list

        if self.save:
            sub_df = pd.DataFrame({'id': test_data['id'],
                                   'score': prediction})
            sub_df['score'] = sub_df['score'].apply(lambda item: int(round(item)))
            sub_df.to_csv('submittion.csv', index=False)

        return oof, prediction

    def _sklearn_tree(self, params):
        train_data, test_data = self._get_dataset()

        columns = train_data.columns

        remove_columns = ['id', 'score']
        features_columns = [column for column in columns if column not in remove_columns]

        train_labels = train_data['score']
        train_x = train_data[features_columns]
        test_x = test_data[features_columns]

        # to csr 加快模型速度
        train_x = sparse.csr_matrix(train_x.values)
        test_x = sparse.csr_matrix(test_x.values)

        kfolder = KFold(n_splits=self.n_fold, shuffle=True, random_state=self.seed)
        kfold = kfolder.split(train_x, train_labels)

        preds_list = list()
        oof = np.zeros(train_data.shape[0])
        for train_index, vali_index in kfold:
            k_x_train = train_x[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_vali = train_x[vali_index]

            gbm = self._get_gbm(params)
            gbm.fit(k_x_train, k_y_train)
            k_pred = gbm.predict(k_x_vali)
            oof[vali_index] = k_pred

            preds = gbm.predict(test_x)
            preds_list.append(preds)

        fold_mae_error = mean_absolute_error(train_labels, oof)
        print(f'{self.mode} fold mae error is {fold_mae_error}')
        fold_score = 1 / (1 + fold_mae_error)
        print(f'fold score is {fold_score}')

        preds_columns = ['preds_{id}'.format(id=i) for i in range(self.n_fold)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        preds_list = list(preds_df.mean(axis=1))
        prediction = preds_list

        if self.save:
            sub_df = pd.DataFrame({'id': test_data['id'],
                                   'score': prediction})
            sub_df['score'] = sub_df['score'].apply(lambda item: int(round(item)))
            sub_df.to_csv('submittion.csv', index=False)

        return oof, prediction

    def _ctb_boost_tree(self, params):
        # catboost 不支持csr，单独考虑
        train_data, test_data = self._get_dataset()

        columns = train_data.columns

        remove_columns = ['id', 'score']
        features_columns = [column for column in columns if column not in remove_columns]

        train_labels = train_data['score']
        train_x = train_data[features_columns]
        test_x = test_data[features_columns]

        kfolder = KFold(n_splits=self.n_fold, shuffle=True, random_state=self.seed)
        kfold = kfolder.split(train_x, train_labels)

        preds_list = list()
        oof = np.zeros(train_data.shape[0])
        for train_index, vali_index in kfold:
            k_x_train = train_x.loc[train_index]
            k_y_train = train_labels.loc[train_index]
            k_x_vali = train_x.loc[vali_index]
            k_y_vali = train_labels.loc[vali_index]

            gbm = self._get_gbm(params)
            gbm = gbm.fit(k_x_train, k_y_train, eval_set=[(k_x_train, k_y_train), (k_x_vali, k_y_vali)],
                          early_stopping_rounds=200, verbose=False)
            iteration_kwargs = self._get_iteration_kwargs(gbm)
            k_pred = gbm.predict(k_x_vali, **iteration_kwargs)
            oof[vali_index] = k_pred

            preds = gbm.predict(test_x, **iteration_kwargs)
            preds_list.append(preds)

        fold_mae_error = mean_absolute_error(train_labels, oof)
        print(f'{self.mode} fold mae error is {fold_mae_error}')
        fold_score = 1 / (1 + fold_mae_error)
        print(f'fold score is {fold_score}')

        preds_columns = ['preds_{id}'.format(id=i) for i in range(self.n_fold)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        preds_list = list(preds_df.mean(axis=1))
        prediction = preds_list

        if self.save:
            sub_df = pd.DataFrame({'id': test_data['id'],
                                   'score': prediction})
            sub_df['score'] = sub_df['score'].apply(lambda item: int(round(item)))
            sub_df.to_csv('submittion.csv', index=False)

        return oof, prediction

    @timer(func_name='TreeModels.tree.model')
    def tree_model(self, params):
        if self.mode in ['lgb', 'xgb']:
            oof, prediction = self._ensemble_tree(params)
        elif self.mode in ['ada', 'rf', 'gbdt']:
            oof, prediction = self._sklearn_tree(params)
        elif self.mode == 'ctb':
            oof, prediction = self._ctb_boost_tree(params)
        else:
            raise ValueError()

        return oof, prediction


def regression_main(mode, **kwargs):
    assert mode in ['lgb', 'xgb', 'rf', 'ctb', 'ada', 'gbdt']
    lgb_params = {
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
        'reg_alpha': 0.15,
        'reg_lambda': 5,
        'verbose': -1,
        'seed': 4590
    }

    # lgb_params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'mae',
    #     'n_estimators': 10000,
    #     'metric': 'mae',
    #     'learning_rate': 0.01,
    #     'min_child_samples': 43,
    #     'min_child_weight': 0,
    #     'subsample_freq': 1,
    #     'num_leaves': 65,
    #     'max_depth': 6,
    #     'subsample': 0.6,
    #     'colsample_bytree': 0.8,
    #     'reg_alpha': 2,
    #     'reg_lambda': 0.1,
    #     'verbose': -1,
    #     'seed': 4590
    # }

    xgb_params = {
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'objective': 'reg:linear',
        'n_estimators': 10000,
        'min_child_weight': 3,
        'gamma': 0,
        'silent': True,
        'n_jobs': 4,
        'random_state': 4590,
        'reg_alpha': 2,
        'reg_lambda': 0.1,
        'alpha': 1,
        'verbose': 1
    }

    ctb_params = {
        'n_estimators': 10000,
        'learning_rate': 0.01,
        'random_seed': 4590,
        'reg_lambda': 5,
        'subsample': 0.7,
        'bootstrap_type': 'Bernoulli',
        'boosting_type': 'Plain',
        'one_hot_max_size': 10,
        'rsm': 0.5,
        'leaf_estimation_iterations': 5,
        'use_best_model': True,
        'max_depth': 6,
        'verbose': -1,
        'thread_count': 4
    }

    gbdt_params = {
        'loss': 'lad',
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'random_state': 2019
    }

    rf_params = {
        'n_estimators': 1000,
        'n_jobs': 5,
        'random_state': 2019
    }

    if mode == 'lgb':
        lgb_oof, lgb_prediction = TreeRegression(mode='lgb', **kwargs).tree_model(lgb_params)
        return lgb_oof, lgb_prediction
    elif mode == 'xgb':
        xgb_oof, xgb_prediction = TreeRegression(mode='xgb', **kwargs).tree_model(xgb_params)
        return xgb_oof, xgb_prediction
    elif mode == 'ctb':
        ctb_oof, ctb_prediction = TreeRegression(mode='ctb', **kwargs).tree_model(ctb_params)
        return ctb_oof, ctb_prediction
    elif mode == 'gbdt':
        gbdt_oof, gbdt_prediction = TreeRegression(mode='gbdt', **kwargs).tree_model(gbdt_params)
        return gbdt_oof, gbdt_prediction
    elif mode == 'rf':
        rf_oof, rf_prediction = TreeRegression(mode='rf', **kwargs).tree_model(rf_params)
        return rf_oof, rf_prediction


if __name__ == '__main__':
    t0 = time.time()
    regression_main(mode='lgb', save=True)
    usage_time = time.time() - t0
    print(f'usage time: {usage_time}')
