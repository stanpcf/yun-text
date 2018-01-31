# coding: utf-8
import argparse
import numpy as np
import pandas as pd

_usage = "ipython -i result_analysis.py h5_file"
_parser = argparse.ArgumentParser(description="预测结果分析工具.", usage=_usage)
_parser.add_argument("h5_path", type=str,
                     help="训练完网络的h5的路径. 例如 bidirectional_lstm/BiLSTM_softmax_50_300_80_2018013014.h5")
_args = _parser.parse_args()

_d, _f = _args.h5_path.split(".")[0].split("/")

valid_prd_file = _d + "/result/valid_" + _f + ".csv"
test_prd_file = _d + "/result/" + _f + ".csv"

valid_prd = pd.read_csv(valid_prd_file, header=None, names=['Id', 'prd'])
test_prd = pd.read_csv(test_prd_file, header=None, names=['Id', 'prd'])

train = pd.read_csv("./input/processed/train_first.csv")
test = pd.read_csv("./input/processed/predict_first.csv")

valid_merge = pd.merge(valid_prd, train, how='left', on=['Id'])
valid_merge = valid_merge[['Id', 'prd', 'Score', 'Discuss_processed', 'Discuss']]
valid_diff = valid_merge[np.abs(valid_merge.prd-valid_merge.Score) > 1]

test_merge = pd.merge(test_prd, test, how='left', on=['Id'])
test_merge = test_merge[['Id', 'prd', 'Discuss_processed', 'Discuss']]
test_low = test_merge[test_merge.prd <= 3]


del _args, _parser, _d, _f, _usage
