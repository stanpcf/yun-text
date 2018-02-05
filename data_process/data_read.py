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


def get_data(train_size=0.8, max_len=80, one_hot=True, return_raw=False, set_cls_weight=False, min_word_len=1,
             cut_tool='all', serial=False):
    """
    :param train_size:
    :param max_len: 给每个句子设定的长度，多截少填
    :param one_hot: 对label进行one-hot编码
    :param return_raw: False则返回tokenizer之后的编码数据. True则返回未编码的数据
    :param set_cls_weight: 是否对训练数据设置权重。
    :param min_word_len: 设置单词的最小长度.
    :param cut_tool: str. 分词工具,如果是多个分词工具, 则下划线连接。 可选[fool, jieba, pynlpir, thulac].
                          为all的时候则是返回所有的分词工具所分的词
    :param serial: 是否将数据串行
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

    fill_value = "CSFxe"
    train = pd.read_csv(os.path.join(data_dir, "processed", "train_first.csv"))
    test = pd.read_csv(os.path.join(data_dir, "processed", "predict_first.csv"))
    train["Discuss"].fillna(value=fill_value, inplace=True)
    train[ctp].fillna(value=fill_value.lower(), inplace=True)
    test["Discuss"].fillna(value=fill_value, inplace=True)
    test[ctp].fillna(value=fill_value.lower(), inplace=True)

    dtrain, dvalid = train_test_split(train, random_state=cfg.TRAIN_TEST_SPLIT_random_state, train_size=train_size)
    x_train_mul, y_train = dtrain[ctp], dtrain["Score"].values - 1
    x_valid_mul, y_valid = dvalid[ctp], dvalid["Score"].values - 1
    x_test_mul = test[ctp]

    result_sentence = OrderedDict()
    if not return_raw:
        tmp_raw = {}
        for tool in ctp:
            x_train = _filter_words(x_train_mul[tool], min_word_len)
            x_valid = _filter_words(x_valid_mul[tool], min_word_len)
            x_test = _filter_words(x_test_mul[tool], min_word_len)

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
        weights = np.array(cfg.SAMPLE_WEIGHT)   # 样本的权重, 索引代表类别, 索引位置的值代表该类别权重 [0, 100, 60, 6, 2, 1]
        index = dtrain['Score'].values - 1
        sample_weights = weights[index]
    else:
        sample_weights = np.ones(dtrain['Score'].shape[0])

    if not serial:
        result = EasyDict({"sent": result_sentence, 'y_train': y_train, 'sample_weights': sample_weights,
                           'y_valid': y_valid, 'valid_id': dvalid['Id'], 'test_id': test["Id"],
                           'tokenizer': tokenizer, 'serial': serial})
    else:
        print("x_train:", result_sentence['fool']['x_train'].shape)
        print("y_train:", y_train.shape)
        _x_train = np.vstack([result_sentence[tool]['x_train'] for tool in ctp])
        _y_train = np.vstack([y_train] * len(ctp))
        _sample_weights = np.hstack([sample_weights] * len(ctp))        # 这个地方使用hstack, 因为sample_weights.shape=(80000,)
        _x_valid = result_sentence[tool]['x_valid']     # 这儿不需要在for里面
        _x_test = result_sentence[tool]['x_test']
        result = EasyDict({"x_train": _x_train, "y_train": _y_train, 'sample_weights': _sample_weights,
                           "x_valid": _x_valid, 'y_valid': y_valid, 'valid_id': dvalid['Id'],
                           "x_test": _x_test, 'test_id': test["Id"],
                           'tokenizer': tokenizer, 'serial': serial})
    return result


def _filter_words(sentences, min_word_len=1):
    if min_word_len == 1:
        return sentences

    result_sentences = []
    for sent in sentences:
        prod = " ".join([w for w in sent.split(" ") if len(w) >= min_word_len])
        result_sentences.append(prod)
    return result_sentences


if __name__ == '__main__':
    data = get_data(min_word_len=1, set_cls_weight=True, cut_tool="all", serial=True)
    print(data.keys())
    print(data.x_train.shape)
    print(data.y_train.shape)
    print(data.sample_weights.shape)
    print(len(data.x_valid))
    print(data.x_valid.shape)
