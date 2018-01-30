# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

MAX_FEATURE = 100000

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")


def get_data(train_size=0.8, max_len=80, one_hot=True, return_raw=False, set_cls_weight=False, min_word_len=2):
    """
    :param train_size:
    :param max_len: 给每个句子设定的长度，多截少填
    :param one_hot: 对label进行one-hot编码
    :param return_raw: False则返回tokenizer之后的编码数据. True则返回未编码的数据
    :param set_cls_weight: 是否对训练数据设置权重。
    :param min_word_len: 设置单词的最小长度.
    :return:
    """
    fill_value = "CSFxe"
    ctp = "Discuss_processed"
    train = pd.read_csv(os.path.join(data_dir, "processed", "train_first.csv"))
    test = pd.read_csv(os.path.join(data_dir, "processed", "predict_first.csv"))
    train["Discuss"].fillna(value=fill_value, inplace=True)
    train[ctp].fillna(value=fill_value.lower(), inplace=True)
    test["Discuss"].fillna(value=fill_value, inplace=True)
    test[ctp].fillna(value=fill_value.lower(), inplace=True)

    dtrain, dvalid = train_test_split(train, random_state=123, train_size=train_size)
    x_train, y_train = dtrain[ctp], dtrain["Score"].values - 1
    x_valid, y_valid = dvalid[ctp], dvalid["Score"].values - 1
    x_test = test[ctp]

    if not return_raw:
        x_train = _filter_words(x_train, min_word_len)
        x_valid = _filter_words(x_valid, min_word_len)
        x_test = _filter_words(x_test, min_word_len)

        tokenizer = Tokenizer(num_words=MAX_FEATURE)
        tokenizer.fit_on_texts(train[ctp].values)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_valid = tokenizer.texts_to_sequences(x_valid)
        x_test = tokenizer.texts_to_sequences(x_test)

        x_train = pad_sequences(x_train, maxlen=max_len)
        x_valid = pad_sequences(x_valid, maxlen=max_len)
        x_test = pad_sequences(x_test, maxlen=max_len)

    if one_hot:
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)

    if set_cls_weight:
        weights = np.array([10, 6, 3, 1.2, 1])   # 样本的权重, 索引代表类别, 索引位置的值代表该类别权重 [0, 100, 60, 6, 2, 1]
        index = dtrain['Score'].values - 1
        sample_weights = weights[index]
    else:
        sample_weights = np.ones(dtrain['Score'].shape[0])

    return (x_train, y_train, sample_weights), (x_valid, y_valid, dvalid['Id']), x_test, test["Id"]


def _filter_words(sentences, min_word_len=1):
    if min_word_len == 1:
        return sentences

    result_sentences = []
    for sent in sentences:
        prod = " ".join([w for w in sent.split(" ") if len(w) >= min_word_len])
        result_sentences.append(prod)
    return result_sentences


if __name__ == '__main__':
    (x_train, y_train, sample_weights), (x_valid, y_valid, valid_id), x_test, test_id = \
        get_data(set_cls_weight=True)
    print(y_train[:10])
    print(sample_weights[:10])
