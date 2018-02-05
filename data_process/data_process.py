#!/usr/bin/env python
# coding: utf8

import os
from tqdm import tqdm
import pandas as pd

# 分词工具
import fool
import jieba
import pynlpir
import thulac

from utils import get_stop_words

stop_words = get_stop_words()


def _filter_stop_words(word_list):
    return [w for w in word_list if w not in stop_words and len(w) > 0]

thu = thulac.thulac(seg_only=True)
pynlpir.open(encoding_errors='ignore')

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
processed_dir = "processed"
train_file = 'train_first.csv'
test_file = 'predict_first.csv'

train = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord', train_file))
test = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord', test_file))

jieba_train, jieba_test = [], []
fool_train, fool_test = [], []
pynlpir_train, pynlpir_test = [], []
thulac_train, thulac_test = [], []

print("process train data")
for text in tqdm(train["Discuss"].values):
    jieba_train.append(" ".join(_filter_stop_words(jieba.cut(text))))
    fool_train.append(" ".join(_filter_stop_words(fool.cut(text)[0])))
    pynlpir_train.append(" ".join(_filter_stop_words(pynlpir.segment(text, pos_tagging=False))))
    thulac_train.append(" ".join(_filter_stop_words([_l[0] for _l in thu.cut(text)])))

print("process test data")
for text in tqdm(test["Discuss"].values):
    jieba_test.append(" ".join(_filter_stop_words(jieba.cut(text))))
    fool_test.append(" ".join(_filter_stop_words(fool.cut(text)[0])))
    pynlpir_test.append(" ".join(_filter_stop_words(pynlpir.segment(text, pos_tagging=False))))
    thulac_test.append(" ".join(_filter_stop_words([_l[0] for _l in thu.cut(text)])))

train['fool'] = fool_train
train['jieba'] = jieba_train
train['pynlpir'] = pynlpir_train
train['thulac'] = thulac_train

test['fool'] = fool_test
test['jieba'] = jieba_test
test['pynlpir'] = pynlpir_test
test['thulac'] = thulac_test

_processed_dir = os.path.join(data_dir, processed_dir)
if not os.path.exists(_processed_dir):
    os.mkdir(_processed_dir)

train.to_csv(os.path.join(data_dir, processed_dir, train_file), index=False)
test.to_csv(os.path.join(data_dir, processed_dir, test_file), index=False)

pynlpir.close()
