#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import inspect
import os
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from utils import get_data
from utils import timer

warnings.filterwarnings(action='ignore')

BASE_PATH = os.path.join('data')
RAWDATA_PATH = os.path.join(BASE_PATH, 'RawData')
ETLDATA_PATH = os.path.join(BASE_PATH, 'EtlData')


class TreeClassfication(object):
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
            gbm = LGBMClassifier(**params)
        elif self.mode == 'xgb':
            gbm = XGBClassifier(**params)
        else:
            raise ValueError()
        return gbm

    @staticmethod
    def _get_label(item):
        if item <= 500:
            return 0
        if item <= 670:
            return 1
        return 2

    def _get_dataset(self):
        dataset = get_data()

        train_data = dataset[dataset['score'] > 0.0]
        test_data = dataset[dataset['score'] < 0.0]

        train_data['label'] = train_data['score'].apply(self._get_label)
        test_data['label'] = -1

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

    @staticmethod
    def _get_preds_label(preds_list):
        preds_shape = preds_list[0].shape[0]
        n_fold = len(preds_list)

        preds = list()
        for i in range(preds_shape):
            preds_item = [0.0, 0.0, 0.0]
            for n in range(n_fold):
                preds_item = [x + y for x, y in zip(preds_item, preds_list[n][i])]
            preds_value = np.argmax(preds_item)
            preds.append(preds_value)

        return preds

    def _ensemble_tree(self, params):
        train_data, test_data = self._get_dataset()

        columns = train_data.columns

        remove_columns = ['id', 'score', 'label']
        features_columns = [column for column in columns if column not in remove_columns]

        train_labels = train_data['label']
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

            preds = gbm.predict_proba(test_x, **iteration_kwargs)
            preds_list.append(preds)

        oof = list(map(int, oof))
        fold_f1_error = f1_score(train_labels.values, oof, average='macro')
        print(f'{self.mode} fold f1 score is {fold_f1_error}')

        prediction = self._get_preds_label(preds_list)

        if self.save:
            test_data['label'] = prediction
            dataset = pd.concat([train_data, test_data], axis=0, ignore_index=True)
            features_name = os.path.join(ETLDATA_PATH, 'features.csv')
            dataset.to_csv(features_name, index=False, encoding='utf-8')

    @timer(func_name='TreeModels.tree.model')
    def tree_model(self, params):
        if self.mode in ['lgb', 'xgb']:
            self._ensemble_tree(params)
        else:
            raise ValueError()


def classfication_main(mode, **kwargs):
    assert mode in ['lgb', 'xgb', 'rf', 'ctb', 'ada', 'gbdt']

    lgb_params = {
        'boosting_type': 'gbdt',
        'n_estimators': 10000,
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.01,
        'min_child_samples': 46,
        'min_child_weight': 0.01,
        'subsample_freq': 20,
        'class_weight': 'balanced',
        'num_leaves': 40,
        'max_depth': 7,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.15,
        'reg_lambda': 5,
        'verbose': -1,
        'seed': 4590
    }

    if mode == 'lgb':
        TreeClassfication(mode='lgb', **kwargs).tree_model(lgb_params)


if __name__ == '__main__':
    t0 = time.time()
    classfication_main(mode='lgb', save=True)
    usage_time = time.time() - t0
    print(f'usage time: {usage_time}')
