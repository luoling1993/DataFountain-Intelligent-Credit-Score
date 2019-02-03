#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold


class Stacking(object):
    def __init__(self, n_fold=10):
        self.n_fold = n_fold

    def get_stacking(self, oof_list, prediction_list, labels):
        train_stack = np.vstack(oof_list).transpose()
        test_stack = np.vstack(prediction_list).transpose()

        repeats = len(oof_list)
        kfolder = RepeatedKFold(n_splits=self.n_fold, n_repeats=repeats, random_state=4590)
        kfold = kfolder.split(train_stack, labels)
        preds_list = list()
        stacking_oof = np.zeros(train_stack.shape[0])

        for train_index, vali_index in kfold:
            k_x_train = train_stack[train_index]
            k_y_train = labels.loc[train_index]
            k_x_vali = train_stack[vali_index]

            gbm = BayesianRidge(normalize=True)
            gbm.fit(k_x_train, k_y_train)

            k_pred = gbm.predict(k_x_vali)
            stacking_oof[vali_index] = k_pred

            preds = gbm.predict(test_stack)
            preds_list.append(preds)

        fold_mae_error = mean_absolute_error(labels, stacking_oof)
        print(f'stacking fold mae error is {fold_mae_error}')
        fold_score = 1 / (1 + fold_mae_error)
        print(f'fold score is {fold_score}')

        preds_columns = ['preds_{id}'.format(id=i) for i in range(self.n_fold * repeats)]
        preds_df = pd.DataFrame(data=preds_list)
        preds_df = preds_df.T
        preds_df.columns = preds_columns
        stacking_prediction = list(preds_df.mean(axis=1))

        return stacking_oof, stacking_prediction
