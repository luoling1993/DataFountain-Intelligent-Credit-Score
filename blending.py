#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error


class Blending(object):
    def __init__(self, train_score_df, num_round=20):
        self.train_score_df = train_score_df
        self.num_round = num_round
        self.score_columns = self._get_score_columns()
        self.score_array = self._get_score_array()

    def _get_score_columns(self):
        df = self.train_score_df.copy()

        remove_columns = ['id', 'score']
        score_columns = list()
        for column in df.columns:
            if column not in remove_columns:
                score_columns.append(column)
        return score_columns

    def _get_score_array(self):
        score_array = list()

        for column in self.score_columns:
            score_array.append(np.array(self.train_score_df[column]))

        return score_array

    def _mae_func(self, weights):
        weight_prediction = 0.0
        y_true = self.train_score_df['score'].values
        for weight, prediction in zip(weights, self.score_array):
            weight_prediction += weight * prediction

        mae_error = mean_absolute_error(y_true, weight_prediction)
        return mae_error

    def get_best_weight(self):
        score_num = len(self.score_columns)
        best_weight = None
        best_error = 9999.9
        for _ in range(self.num_round):
            weight = np.random.dirichlet(alpha=np.ones(score_num), size=1).flatten()
            bounds = [(0, 1)] * score_num

            res = minimize(self._mae_func, weight, method='L-BFGS-B', bounds=bounds,
                           options={'disp': False, 'maxiter': 100000})
            res_error = res['fun']

            if res_error < best_error:
                best_error = res_error
                best_weight = res['x']

        # 归一化
        best_weight_sum = np.sum(best_weight)
        best_weight = best_weight / best_weight_sum

        print(f'best mae error {best_error}')
        mae_score = 1 / (1 + best_error)
        print(f'best mae score: {mae_score}')
        print(best_weight)
        return best_weight
