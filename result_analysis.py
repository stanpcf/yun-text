# coding: utf-8
import numpy as np
import pandas as pd
import os
from collections import namedtuple

"""
usage: ipython -i result_analysis.py
"""
pro_dir = os.path.dirname(os.path.abspath(__file__))
train = pd.read_csv(pro_dir + "/input/processed/train_first.csv")
test = pd.read_csv(pro_dir + "/input/processed/predict_first.csv")


def get_result_stat(h5_path):
    """
    :param h5_path: 训练完之后的h5_file路径
    bidirectional_lstm/BiLSTM_softmax_50_300_80_2018013014.h5
    """
    _d, _f = h5_path.split(".")[0].split("/")

    valid_prd_file = _d + "/result/valid_" + _f + ".csv"
    test_prd_file = _d + "/result/" + _f + ".csv"

    valid_prd = pd.read_csv(valid_prd_file, header=None, names=['Id', 'prd'])
    test_prd = pd.read_csv(test_prd_file, header=None, names=['Id', 'prd'])

    valid_merge = pd.merge(valid_prd, train, how='left', on=['Id'])
    valid_merge = valid_merge[['Id', 'prd', 'Score', 'fool', 'Discuss']]
    valid_diff = valid_merge[np.abs(valid_merge.prd-valid_merge.Score) > 1]

    test_merge = pd.merge(test_prd, test, how='left', on=['Id'])
    test_merge = test_merge[['Id', 'prd', 'fool', 'Discuss']]
    test_low = test_merge[test_merge.prd <= 3]
    ResultStat = namedtuple("ResultStat", ["valid_merge", "valid_diff", "test_merge", "test_low", "test_prd"])
    result = ResultStat(valid_merge=valid_merge, valid_diff=valid_diff, test_merge=test_merge, test_low=test_low,
                        test_prd=test_prd)
    return result
