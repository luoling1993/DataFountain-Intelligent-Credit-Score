#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')


class Selector(object):
    def __init__(self, ascending=False):
        self.ascending = ascending

    @staticmethod
    def _get_xgb_model(**kwargs):
        xgb_params = {
            'booster': 'gbtree',
            'learning_rate': 0.05,
            'objective': 'reg:linear',
            'n_estimators': 10000,
            'silent': True,
            'n_jobs': 4,
            'random_state': 4590,
            'verbose': -1
        }
        for k, v in kwargs.items():
            xgb_params[k] = v
        xgb_model = xgb.XGBRegressor(**xgb_params)

        return xgb_model

    @staticmethod
    def _get_lgb_model(**kwargs):
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'mae',
            'n_estimators': 10000,
            'metric': 'mae',
            'learning_rate': 0.05,
            'n_jobs': 4,
            'verbose': -1,
            'seed': 4590
        }
        for k, v in kwargs.items():
            lgb_params[k] = v

        lgb_model = lgb.LGBMRegressor(**lgb_params)

        return lgb_model

    @staticmethod
    def _get_importance_features(model, columns, topn=300, ascending=False):
        # ascending=False: 降序
        # ascending=False: 升序
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({'column': columns, 'score': feature_importance})
        importance_df = importance_df.sort_values(by=['score'], ascending=ascending).reset_index()
        importance_columns = importance_df['column'].loc[:topn].tolist()
        return importance_columns

    def _get_cv_error(self, x_train, y_train):
        # model_list = [self._get_xgb_model(), self._get_lgb_model()]
        model_list = [self._get_lgb_model()]  # xgb速度过于慢，暂时只用lgb尝试

        cv_error = 0.0
        mae = make_scorer(mean_absolute_error)
        for model in model_list:
            mae_error = cross_val_score(model, x_train, y_train, scoring=mae, cv=5, n_jobs=5)
            mae_error = np.mean(mae_error)
            cv_error += mae_error

        model_length = len(model_list)
        cv_error = cv_error / model_length
        return cv_error

    def _get_increase_features(self, x_train, y_train):
        # 获取有用特征
        columns = x_train.columns

        importance_columns_list = list()
        for model in [self._get_xgb_model(), self._get_lgb_model()]:
            meta_model = model.fit(x_train, y_train)
            model_importance_columns = self._get_importance_features(meta_model, columns)
            importance_columns_list.extend(model_importance_columns)

        columns_num = 1
        select_columns = list()
        cv_error = 999.0
        importance_columns_set = set()

        for index, column in enumerate(importance_columns_list):
            if column in importance_columns_set:
                # set function will upset importance_columns_list order
                continue
            else:
                importance_columns_set.add(column)

            select_columns.append(column)
            x_train_sample = x_train[select_columns]
            tmp_cv_error = self._get_cv_error(x_train_sample, y_train)
            if tmp_cv_error < cv_error:
                cv_error = tmp_cv_error
                print(f'columns_num:{columns_num}\tindex_num:{index}\tcv_error:{cv_error}\tcolumn:{column}')
                columns_num += 1
            else:
                print(f'ignore column: {column}\tindex_num:{index}\tcv_error: {tmp_cv_error}')
                select_columns.pop()

        return select_columns

    def _get_reduce_features(self, x_train, y_train):
        # 删除无用特征
        columns = x_train.columns

        importance_columns_list = list()
        for model in [self._get_xgb_model(), self._get_lgb_model()]:
            meta_model = model.fit(x_train, y_train)
            model_importance_columns = self._get_importance_features(meta_model, columns, ascending=True)
            importance_columns_list.extend(model_importance_columns)

        columns_num = x_train.shape[1]
        select_columns = list(x_train.columns)
        cv_error = 999.0
        unimportance_columns_set = set()

        for index, column in enumerate(importance_columns_list):
            if column in unimportance_columns_set:
                # set function will upset importance_columns_list order
                continue
            else:
                unimportance_columns_set.add(column)

            select_columns.remove(column)
            x_train_sample = x_train[select_columns]
            tmp_cv_error = self._get_cv_error(x_train_sample, y_train)
            if tmp_cv_error < cv_error:
                cv_error = tmp_cv_error
                print(f'columns_num:{columns_num}\tindex_num:{index}\tcv_error:{cv_error}\tcolumn:{column}')
                columns_num -= 1
            else:
                print(f'stay column: {column}\tindex_num:{index}\tcv_error: {tmp_cv_error}')
                select_columns.append(column)

        return select_columns

    def get_select_features(self, x_train, y_train):
        if self.ascending:
            return self._get_reduce_features(x_train, y_train)
        else:
            return self._get_increase_features(x_train, y_train)




