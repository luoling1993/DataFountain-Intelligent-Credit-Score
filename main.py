#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

from blending import Blending
# from processing import processing_main
from stacking import Stacking
from tree_regression import regression_main
from utils import get_blending_score
from utils import get_combinations
from utils import get_data
from utils import get_ensemble_score
from utils import get_score_array
from utils import get_values_by_index

if __name__ == '__main__':
    t0 = time.time()
    """
    selectot=True 使用selector抽取部分特征
    但是速度慢且效果不好，暂时放弃
    """
    # processing_main(selector=False)  # 若以生成特征，可注释
    dataset = get_data()

    train_data = dataset[dataset['score'] > 0.0]
    test_data = dataset[dataset['score'] < 0.0]

    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)

    train_score_df = train_data[['id', 'score']]
    test_score_df = test_data[['id']]

    oof_list = list()
    prediction_list = list()

    # AdaBoost效果不好
    # rf也会降分，而且非常慢
    # mode_list = ['lgb', 'xgb', 'ctb', 'gbdt']
    mode_list = ['lgb', 'xgb', 'ctb']
    # mode_list = ['lgb']
    for mode in mode_list:
        mode_score_name = f'{mode}_score'
        oof, prediction = regression_main(mode=mode)
        oof_list.append(oof)
        prediction_list.append(prediction)

        train_score = oof.tolist()
        train_score_df[mode_score_name] = train_score
        test_score_df[mode_score_name] = prediction

    # stacking
    combinations_list = get_combinations(range(len(oof_list) + 1))
    for bin_item in combinations_list:
        oof = get_values_by_index(oof_list, bin_item)
        prediction = get_values_by_index(prediction_list, bin_item)

        mode = get_values_by_index(mode_list, bin_item)
        mode.append('score')
        mode_name = '_'.join(mode)

        stacking_oof, stacking_prediction = Stacking().get_stacking(oof, prediction, train_score_df['score'])
        train_score_df[mode_name] = stacking_oof
        test_score_df[mode_name] = stacking_prediction

    # 加权分数
    get_ensemble_score(train_score_df)

    # blending
    best_weight = Blending(train_score_df).get_best_weight()
    score_array = get_score_array(test_score_df)
    test_score_df['score'] = get_blending_score(score_array, best_weight)
    test_score_df.to_csv('all_score.csv', index=False)

    sub_df = test_score_df[['id', 'score']]
    sub_df['score'] = sub_df['score'].apply(lambda item: int(round(item)))
    sub_df.to_csv('submittion.csv', index=False)

    usage_time = time.time() - t0
    print(f'usage time: {usage_time}')
