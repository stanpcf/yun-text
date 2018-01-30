# -*- coding: utf-8 -*-

import os
from abc import ABCMeta, abstractmethod
from datetime import datetime

import pandas as pd
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K

from metric import yun_metric


class TextModel(object):
    """abstract base model for all text classification model."""
    __metaclass__ = ABCMeta

    def __init__(self, nb_epoch=50, max_len=100, embed_size=100, last_act='softmax', batch_size=640, optimizer='adam',
                 use_pretrained=False, trainable=True, min_word_len=2, **kwargs):
        """
        :param nb_epoch: 迭代次数
        :param max_len:  规整化每个句子的长度
        :param embed_size: 词向量维度
        :param last_act: 最后一层的激活函数
        :param batch_size:
        :param optimizer: 优化器
        :param use_pretrained: 是否嵌入层使用预训练的模型
        :param trainable: 是否嵌入层可训练, 该参数只有在use_pretrained为真时有用
        :param min_word_len: 同data_process/get_data的min_word_len存储信息一样. 用于生成weight_path
        :param kwargs: dict: global_data为true, 则必须包含
                (x_train, y_train, x_valid, y_valid, x_test, test_id) 这几项
        """
        self.nb_epoch = nb_epoch
        self.max_len = max_len
        self.embed_size = embed_size
        self.last_act = last_act
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.use_pretrained = use_pretrained
        self.trainable = trainable
        self.min_word_len = min_word_len
        self.time = datetime.now().strftime('%Y%m%d%H')

        self.x_train = kwargs['x_train']
        self.y_train = kwargs['y_train']
        self.sample_weights = kwargs['sample_weights']
        self.x_valid = kwargs['x_valid']
        self.y_valid = kwargs['y_valid']
        self.valid_id = kwargs['valid_id']
        self.x_test = kwargs['x_test']
        self.test_id = kwargs['test_id']
        assert self.max_len == self.x_train.shape[-1]

    @abstractmethod
    def get_model(self) -> Model:
        """定义一个keras net, compile it and return the model"""
        raise NotImplementedError

    @abstractmethod
    def _get_bst_model_path(self) -> str:
        """return a name which is used for save trained weights"""
        raise NotImplementedError

    def get_bst_model_path(self):
        dirname = self._get_model_path()
        path = os.path.join(dirname, self._get_bst_model_path())
        return path

    def train(self):
        model = self.get_model()
        model.summary()

        bst_model_path = self.get_bst_model_path()
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        callback_list = [model_checkpoint, early_stopping]
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.nb_epoch,
                  validation_split=0.1, callbacks=callback_list, sample_weight=self.sample_weights)
        print("model train finish: ", bst_model_path, "at time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def predict(self, predict_offline=True, bst_model_path=None):
        if not bst_model_path:
            bst_model_path = self.get_bst_model_path()

        model = self.get_model()
        model.load_weights(bst_model_path)
        if predict_offline:
            _valid_pred = model.predict(self.x_valid)
            valid_pred = _valid_pred.argmax(axis=1)
            valid_pred = valid_pred + 1
            self._save_to_csv(self.valid_id, valid_pred, bst_model_path, valid_data=True)
            _y_valid = self.y_valid.argmax(axis=1) + 1
            print("valid yun metric:", yun_metric(_y_valid, valid_pred))
        _y_test = model.predict(self.x_test)
        y_test = _y_test.argmax(axis=1)
        y_test = y_test + 1
        self._save_to_csv(self.test_id, y_test, bst_model_path, valid_data=False)

    def _save_to_csv(self, ids, scores, path, valid_data=False):
        assert len(ids) == len(scores)
        sample_submission = pd.DataFrame({
            "Id": ids,
            "Score": scores
        })
        if not valid_data:
            result_path = self._get_result_path(path)
        else:
            _tmp = self._get_result_path(path)
            _list = _tmp.split("/")
            _tmp = _list[:-1] + ["valid_"+_list[-1]]
            result_path = "/".join(_tmp)
        sample_submission.to_csv(result_path, index=False, header=False)

    def _get_result_path(self, bst_model_path):
        basename = os.path.basename(bst_model_path)
        dirname = os.path.dirname(bst_model_path)
        dirname = os.path.join(dirname, 'result')

        if not os.path.exists(dirname):
            os.mkdir(path=dirname)
        result = os.path.join(dirname, basename[:-3]+'.csv')
        return result

    def _get_model_path(self):
        _module = self.__class__.__dict__.get('__module__')
        model_dir = "/".join(_module.split(".")[:-1])
        return model_dir


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim
