#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from selector import Selector
from utils import timer

warnings.filterwarnings(action='ignore')

BASE_PATH = os.path.join('data')
RAWDATA_PATH = os.path.join(BASE_PATH, 'RawData')
ETLDATA_PATH = os.path.join(BASE_PATH, 'EtlData')


class Processing(object):
    def __init__(self, selector=False, ascending=False):
        self.selector = selector
        self.ascending = ascending

    @staticmethod
    def _get_columns_name():
        columns_name = ['id', 'is_real_name', 'age', 'is_college_student', 'is_blacklist', 'is_illbeing_4g',
                        'surfing_time', 'last_pay_month', 'last_pay_acount', 'avg_pay_acount', 'this_month_acount',
                        'this_month_balance', 'is_arrearage', 'acount_sensitivity', 'this_month_call_num',
                        'is_shopping', 'avg_shopping_num', 'is_wanda', 'is_sam', 'is_movie', 'is_travel', 'is_sports',
                        'online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                        'train_num', 'travel_num', 'score']

        return columns_name

    def _get_data(self):
        columns_name = self._get_columns_name()

        train_data_name = os.path.join(RAWDATA_PATH, 'train_dataset.csv')
        train_data = pd.read_csv(train_data_name, header=0)
        train_data.columns = columns_name

        test_data_name = os.path.join(RAWDATA_PATH, 'test_dataset.csv')
        test_data = pd.read_csv(test_data_name, header=0)
        test_data['score'] = -1
        test_data.columns = columns_name

        dataset = pd.concat([train_data, test_data], ignore_index=True)
        return dataset

    @staticmethod
    def _get_boolean_columns(dataset):
        dataset = dataset.copy()
        boolean_columns = list()

        for column in dataset.columns:
            nunique = dataset[column].nunique()
            if nunique == 2:
                boolean_columns.append(column)

        return boolean_columns

    @staticmethod
    def _get_missing_value(item):
        if item == 0:
            return 1
        return 0

    @staticmethod
    def _get_abnormal_label(item):
        if item == 0:
            return 0
        if item < 10:
            return 1
        if item < 100:
            return 2
        if item < 1000:
            return 3
        else:
            return 4

    @staticmethod
    def _recombine_boolean_columns(dataset, boolean_columns):
        # 通过2进制编码
        dataset = dataset.copy()

        bin_base = 1
        dataset['boolean_bin'] = 0  # 初始化
        for column in boolean_columns:
            dataset['boolean_bin'] += dataset[column] * bin_base
            bin_base = 2 * bin_base

        # 有些组合出现次数过少，合并
        counter = Counter(dataset['boolean_bin'])
        counter_dict = dict()
        for item, count in counter.items():
            if count < 5:
                counter_dict[item] = -1
            else:
                counter_dict[item] = count
        dataset['boolean_bin'] = dataset['boolean_bin'].map(counter_dict)

        # One-Hot
        dataset = pd.get_dummies(dataset, columns=['boolean_bin'])
        return dataset

    @staticmethod
    def _get_recharge_way(item):
        # 是否能被10整除
        if item == 0:
            return -1
        if item % 10 == 0:
            return 1
        else:
            return 0

    @staticmethod
    def _shopping_encoder(item):
        is_shopping = item['is_shopping']
        avg_shopping_num = item['avg_shopping_num']

        if is_shopping == 0:
            if avg_shopping_num < 10:
                return 0
            elif avg_shopping_num < 20:
                return 1
            else:
                return 2
        else:
            if avg_shopping_num < 20:
                return 3
            return 4

    @staticmethod
    def _get_app_rate(dataset):
        dataset = dataset.copy()

        app_num_columns = ['online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                           'train_num', 'travel_num']
        dataset['helper_sum'] = dataset[app_num_columns].apply(lambda item: np.log1p(np.sum(item)), axis=1)

        for column in app_num_columns:
            column_name = f'{column}_rate'
            dataset[column_name] = np.log1p(dataset[column]) / dataset['helper_sum']

        # dataset = dataset.drop(columns=['helper_sum'])
        return dataset

    def _get_operation_features(self, dataset):
        dataset = dataset.copy()

        dataset['recharge_way'] = dataset['last_pay_acount'].apply(self._get_recharge_way)
        dataset = pd.get_dummies(dataset, columns=['recharge_way'])

        # 稳定性
        # 当月话费 / (近6个月平均话费 + 5)
        dataset['month_half_year_stable'] = dataset['this_month_acount'] / (dataset['avg_pay_acount'] + 5)
        dataset['month_half_year_diff'] = dataset['this_month_acount'] - dataset['avg_pay_acount']
        # 当月话费 / (当月账户余额 + 5)
        dataset['use_left_stable'] = dataset['this_month_acount'] / (dataset['this_month_balance'] + 5)
        dataset['use_left_diff'] = dataset['this_month_acount'] - dataset['this_month_balance']

        # 商场行为编码
        dataset['shopping_encoder'] = dataset[['is_shopping', 'avg_shopping_num']].apply(self._shopping_encoder, axis=1)
        dataset = pd.get_dummies(dataset, columns=['shopping_encoder'])

        # 上网时长
        dataset['surfing_time_copy'] = dataset['surfing_time']
        dataset['surfing_time_copy'] = pd.qcut(dataset['surfing_time_copy'], 5, labels=False)
        dataset = pd.get_dummies(dataset, columns=['surfing_time_copy'])

        # APP打开占比
        dataset = self._get_app_rate(dataset)

        return dataset

    @staticmethod
    def _data_encoder(dataset, num_columns):
        dataset = dataset.copy()

        # # LabelEncoder
        # for column in num_columns:
        #     mapping_dict = dict(zip(dataset[column].unique(), range(0, dataset[column].nunique())))
        #     dataset[column] = dataset[column].map(mapping_dict)
        # qcut
        for column in num_columns:
            dataset[column] = pd.qcut(dataset[column], 20, labels=False, duplicates='drop')

        train_data = dataset[dataset['score'] > 0]
        train_data['helper'] = pd.cut(train_data['score'], 5, labels=False)
        train_data = pd.get_dummies(train_data, columns=['helper'])
        helper_columns = ['helper_0', 'helper_1', 'helper_2', 'helper_3', 'helper_4']

        for column in num_columns:
            for helper_column in helper_columns:
                column_name = f'{column}_{helper_column}_mean'
                column_df = train_data.groupby(by=[column])[helper_column].agg('mean').reset_index(name='mean')
                column_dict = column_df.set_index(column)['mean'].to_dict()

                dataset[column_name] = dataset[column].map(column_dict)

        return dataset

    @timer(func_name='Processing.get_processing')
    def get_processing(self):
        dataset = self._get_data()

        boolean_columns = self._get_boolean_columns(dataset)
        remove_columns = ['id', 'score']
        num_columns = list()
        for column in dataset.columns:
            if column in remove_columns:
                continue
            if column in boolean_columns:
                continue
            num_columns.append(column)

        # 异常字段处理：手动分箱
        abnormal_columns = ['online_shopping_num', 'logistics_num', 'financing_num', 'video_num', 'airplant_num',
                            'train_num', 'travel_num']
        abnormal_encoder_columns = list()
        for column in abnormal_columns:
            encoder_column = f'{column}_encoder'
            dataset[encoder_column] = dataset[column].apply(self._get_abnormal_label)
            abnormal_encoder_columns.append(encoder_column)
        dataset = pd.get_dummies(dataset, columns=abnormal_encoder_columns)  # One-Hot

        # 缺失值单独抽离特征：无效
        for column in num_columns:
            # abnormal已处理过，continue
            if column in abnormal_columns:
                continue
            column_name = f'{column}_missing'
            dataset[column_name] = dataset[column].apply(self._get_missing_value)

        # 将bool类型重新组合
        # dataset = self._recombine_boolean_columns(dataset, boolean_columns)

        # embedding
        # dataset = self._data_encoder(dataset, ['surfing_time', 'age'])

        # 业务逻辑特征
        dataset = self._get_operation_features(dataset)

        if self.selector:
            train_data = dataset[dataset['score'] > 0]
            y_data = train_data['score']
            x_data = train_data.drop(columns=['id', 'score'])
            select_fectures = Selector(ascending=self.ascending).get_select_features(x_data, y_data)
            select_fectures.extend(['id', 'score'])
            dataset = dataset[select_fectures]

        return dataset


def processing_main(selector=False, ascending=False):
    processing = Processing(selector=selector, ascending=ascending)
    dt = processing.get_processing()

    features_name = os.path.join(ETLDATA_PATH, 'features.csv')
    dt.to_csv(features_name, index=False, encoding='utf-8')


if __name__ == '__main__':
    processing_main(selector=False, ascending=False)
