# -*- coding: utf-8 -*-

import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

MAX_FEATURE = 100000

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")


def get_data(train_size=0.8, max_len=80, one_hot=True):
    """
    :param train_size:
    :param max_len: 给每个句子设定的长度，多截少填
    :return: x.shape=(sample, max_len), every sentence is sequenced.

    """
    fill_value = "CSFxe"
    ctp = "Discuss_processed"
    train = pd.read_csv(os.path.join(data_dir, "processed", "train_first.csv"))
    test = pd.read_csv(os.path.join(data_dir, "processed", "predict_first.csv"))
    train["Discuss"].fillna(value=fill_value, inplace=True)
    train[ctp].fillna(value=fill_value.lower(), inplace=True)
    test["Discuss"].fillna(value=fill_value, inplace=True)
    test[ctp].fillna(value=fill_value.lower(), inplace=True)

    tokenizer = Tokenizer(num_words=MAX_FEATURE)
    tokenizer.fit_on_texts(train[ctp].values)

    dtrain, dvalid = train_test_split(train, random_state=123, train_size=train_size)
    x_train, y_train = dtrain[ctp], dtrain["Score"].values
    x_valid, y_valid = dvalid[ctp], dvalid["Score"].values
    x_test = test[ctp]

    x_train = tokenizer.texts_to_sequences(x_train)
    x_valid = tokenizer.texts_to_sequences(x_valid)
    x_test = tokenizer.texts_to_sequences(x_test)

    x_train = pad_sequences(x_train, maxlen=max_len)
    x_valid = pad_sequences(x_valid, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    if one_hot:
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)

    return (x_train, y_train), (x_valid, y_valid, dvalid['Id']), x_test, test["Id"]


if __name__ == '__main__':
    (x_train, y_train), (x_valid, y_valid, valid_id), x_test, test_id = get_data()
