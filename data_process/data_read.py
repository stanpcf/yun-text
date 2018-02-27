# -*- coding: utf-8 -*-

import os
from easydict import EasyDict
from collections import OrderedDict
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

try:
    from .config import cfg
except ModuleNotFoundError as e:
    from config import cfg

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")


def get_data(train_size=0.8, max_len=80, one_hot=True, return_raw=False, set_cls_weight=False,
             cut_tool='all', serial=False, cls_weights_str=None, num_class=5):
    """
    :param train_size:
    :param max_len: 给每个句子设定的长度，多截少填
    :param one_hot: 对label进行one-hot编码
    :param return_raw: False则返回tokenizer之后的编码数据. True则返回未编码的数据
    :param set_cls_weight: 是否对训练数据设置权重。
    :param cut_tool: str. 分词工具,如果是多个分词工具, 则下划线连接。 可选[fool, jieba, pynlpir, thulac].
                          为all的时候则是返回所有的分词工具所分的词
    :param serial: 是否将数据串行
    :param cls_weights_str: 权重字符串
    :param num_class:
    :return: EasyDict实例.
        并行的key的结构为
            sent: 该key下面的key是可选的
                fool: x_train, x_valid, x_test
                jieba: x_train, x_valid, x_test
                pynlpir: x_train, x_valid, x_test
                thulac: x_train, x_valid, x_test
            y_train
            sample_weights
            y_valid
            valid_id
            test_id
            tokenizer
            serial
        串行的key的结构为
            x_train, y_train, sample_weights, x_valid, y_valid, valid_id, x_test, test_id, tokenizer, serial
    """
    all = ['fool', 'jieba', 'pynlpir', 'thulac']
    if cut_tool != 'all':
        ctp = cut_tool.split('_')
        for c in ctp:
            assert c in all
    else:
        ctp = all

    if not cls_weights_str:
        class_weights = [100, 60, 6, 2, 1]
    else:
        class_weights = [float(i) for i in cls_weights_str.split("_")]
        assert len(class_weights) == 5

    if num_class == 6:
        class_weights = [0] + class_weights

    fill_value = "CSFxe"
    train = pd.read_csv(os.path.join(data_dir, "processed", "train_first.csv"))
    test = pd.read_csv(os.path.join(data_dir, "processed", "predict_first.csv"))
    train["Discuss"].fillna(value=fill_value, inplace=True)
    test["Discuss"].fillna(value=fill_value, inplace=True)
    for tool in ctp:
        train[tool].fillna(value=fill_value.lower(), inplace=True)
        test[tool].fillna(value=fill_value.lower(), inplace=True)

    dtrain, dvalid = train_test_split(train, random_state=cfg.TRAIN_TEST_SPLIT_random_state, train_size=train_size)
    x_train_mul = dtrain[ctp]
    x_valid_mul = dvalid[ctp]
    x_test_mul = test[ctp]

    if one_hot:
        if num_class == 5:
            y_train = dtrain["Score"].values - 1
            y_valid = dvalid["Score"].values - 1
        elif num_class == 6:
            y_train = dtrain["Score"].values
            y_valid = dvalid["Score"].values
        else:
            raise Exception
    else:
        y_train = dtrain["Score"].values
        y_valid = dvalid["Score"].values

    result_sentence = OrderedDict()
    if not return_raw:
        tmp_raw = {}
        for tool in ctp:
            x_train = x_train_mul[tool]
            x_valid = x_valid_mul[tool]
            x_test = x_test_mul[tool]

            tmp_raw[tool] = (x_train, x_valid, x_test)

        corpus = []
        for tn in tmp_raw.values():
            corpus.extend(tn[0])
            corpus.extend(tn[1])
            corpus.extend(tn[2])
        tokenizer = Tokenizer(num_words=cfg.MAX_FEATURE)
        tokenizer.fit_on_texts(corpus)

        for tool in ctp:
            tn = tmp_raw[tool]
            x_train = tokenizer.texts_to_sequences(tn[0])
            x_valid = tokenizer.texts_to_sequences(tn[1])
            x_test = tokenizer.texts_to_sequences(tn[2])

            x_train = pad_sequences(x_train, maxlen=max_len, padding=cfg.Keras_padding)
            x_valid = pad_sequences(x_valid, maxlen=max_len, padding=cfg.Keras_padding)
            x_test = pad_sequences(x_test, maxlen=max_len, padding=cfg.Keras_padding)
            result_sentence[tool] = {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test}
    else:
        tokenizer = None
        for tool in ctp:
            x_train = x_train_mul[tool]
            x_valid = x_valid_mul[tool]
            x_test = x_test_mul[tool]
            result_sentence[tool] = {'x_train': x_train, 'x_valid': x_valid, 'x_test': x_test}

    if one_hot:
        y_train = to_categorical(y_train)
        y_valid = to_categorical(y_valid)

    if set_cls_weight:
        weights = np.array(class_weights)
        if num_class == 5:
            index = dtrain['Score'].values - 1
        elif num_class == 6:
            index = dtrain['Score'].values
        else:
            raise Exception
        print("Class weights: ", weights)
        sample_weights = weights[index]
    else:
        sample_weights = np.ones(dtrain['Score'].shape[0])

    if not serial:
        result = EasyDict({"sent": result_sentence, 'y_train': y_train, 'sample_weights': sample_weights,
                           'y_valid': y_valid, 'valid_id': dvalid['Id'].values, 'test_id': test["Id"].values,
                           'tokenizer': tokenizer, 'serial': serial})
    else:
        _x_train = np.vstack([result_sentence[tool]['x_train'] for tool in ctp])
        if one_hot:
            _y_train = np.vstack([y_train] * len(ctp))
        else:
            _y_train = np.hstack([y_train] * len(ctp))
        _sample_weights = np.hstack([sample_weights] * len(ctp))        # 这个地方使用hstack, 因为sample_weights.shape=(80000,)
        _x_valid = result_sentence[tool]['x_valid']     # 这儿不需要在for里面
        _x_test = result_sentence[tool]['x_test']
        result = EasyDict({"x_train": _x_train, "y_train": _y_train, 'sample_weights': _sample_weights,
                           "x_valid": _x_valid, 'y_valid': y_valid, 'valid_id': dvalid['Id'].values,
                           "x_test": _x_test, 'test_id': test["Id"].values,
                           'tokenizer': tokenizer, 'serial': serial})
    return result


if __name__ == '__main__':
    data = get_data(set_cls_weight=False, cut_tool="fool", serial=True, one_hot=False)
    print(data.keys())
    print(data.x_train.shape)
    print(data.y_train.shape)
    print(data.sample_weights.shape)
    print(len(data.x_valid))
    print(data.x_valid.shape)
