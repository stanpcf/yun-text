#!/usr/bin/env python
# coding: utf8

import os
import pandas as pd
import jieba

from utils import process_str, fill_value

user_dict = './yan_word.txt'

jieba.load_userdict(user_dict)

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
processed_dir = "processed"
train_file = 'train_second.csv'
test_file = 'predict_second.csv'

train_a = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord', 'train_first.csv'))
train_b = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord-Semi', train_file))
train = pd.concat([train_a, train_b], ignore_index=True)
print("--->train shape:", train.shape)
test_a = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord', 'predict_first.csv'))
test = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord-Semi', test_file))

jieba_train, jieba_test = [], []

print("process data. need about half an hour")
train['jieba'] = train['Discuss'].map(lambda x: process_str(x))
test['jieba'] = test['Discuss'].map(lambda x: process_str(x))
test_a['jieba'] = test_a['Discuss'].map(lambda x: process_str(x))

for col in ['Discuss', 'jieba']:
    train[col].fillna(value=fill_value, inplace=True)
    test[col].fillna(value=fill_value, inplace=True)
    test_a[col].fillna(value=fill_value, inplace=True)

_processed_dir = os.path.join(data_dir, processed_dir)
if not os.path.exists(_processed_dir):
    os.mkdir(_processed_dir)

train.to_csv(os.path.join(data_dir, processed_dir, train_file), index=False)
test.to_csv(os.path.join(data_dir, processed_dir, test_file), index=False)
test_a.to_csv(os.path.join(data_dir, processed_dir, 'predict_first.csv'), index=False)
