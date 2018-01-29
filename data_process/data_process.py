#!/usr/bin/env python
# coding: utf8

import os
import fool
from tqdm import tqdm
import pandas as pd

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
processed_dir = "processed"
train_file = 'train_first.csv'
test_file = 'predict_first.csv'

train = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord', train_file))
test = pd.read_csv(os.path.join(data_dir, 'YNU.EDU2018-ScenicWord', test_file))

Discuss_processed_train = []
Discuss_processed_test = []
for text in tqdm(train["Discuss"].values):
    tokens = fool.cut(text)[0]
    Discuss_processed_train.append(" ".join(tokens))

for text in tqdm(test["Discuss"].values):
    tokens = fool.cut(text)[0]
    Discuss_processed_test.append(" ".join(tokens))


train['Discuss_processed'] = Discuss_processed_train
test['Discuss_processed'] = Discuss_processed_test

if not os.path.exists(processed_dir):
    os.mkdir(os.path.join(data_dir, processed_dir))

train.to_csv(os.path.join(data_dir, processed_dir, train_file))
test.to_csv(os.path.join(data_dir, processed_dir, test_file))
