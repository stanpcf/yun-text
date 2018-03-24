# -*- coding: utf-8 -*-
import os
from abc import ABCMeta, abstractmethod
from datetime import datetime

import pandas as pd
import numpy as np
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from keras import backend as K
from sklearn.model_selection import StratifiedKFold, train_test_split

from metric import yun_metric
from data_process import cfg


class TextModel(object):
    """abstract base model for all text classification model."""
    __metaclass__ = ABCMeta

    def __init__(self, *, data, nb_epoch=50, max_len=100, embed_size=100, batch_size=640,
                 optimizer='adam', use_pretrained=False, trainable=True, **kwargs):
        """
        :param data: data_process.get_data返回的对象
        :param nb_epoch: 迭代次数
        :param max_len:  规整化每个句子的长度
        :param embed_size: 词向量维度
        :param last_act: 最后一层的激活函数
        :param batch_size:
        :param optimizer: 优化器
        :param use_pretrained: 是否嵌入层使用预训练的模型
        :param trainable: 是否嵌入层可训练, 该参数只有在use_pretrained为真时有用
        :param kwargs
        """
        self.nb_epoch = nb_epoch
        self.max_len = max_len
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.use_pretrained = use_pretrained
        self.trainable = trainable
        self.time = datetime.now().strftime('%y%m%d%H%M%S')
        # self.callback_list = []
        self.kwargs = kwargs
        self.data = data
        self.is_kfold = kwargs.get('is_kfold', False)
        self.kfold = kwargs.get('kfold', 0)
        if self.is_kfold:
            self.bst_model_path_list = []
        self.is_retrain = kwargs.get('is_retrain') if not self.trainable else False  # 当trainble 为False时才is_retrain 可用

    @abstractmethod
    def get_model(self, trainable=None) -> Model:
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
        bst_model_path = self.get_bst_model_path()
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        if self.is_kfold:
            test_prd_mean = []
            folds = list(StratifiedKFold(n_splits=self.kfold, shuffle=True,
                                         random_state=2017).split(self.data.x_train, self.data.y_train))
            for i, (train_index, valid_index) in enumerate(folds, start=1):
                model = self.get_model()
                model.summary()
                bmp = bst_model_path + '_' + str(i)
                self.bst_model_path_list.append(bmp)
                model_checkpoint = ModelCheckpoint(bmp, save_best_only=False, save_weights_only=True)
                x_train, x_valid = self.data.x_train[train_index], self.data.x_train[valid_index]
                y_train, y_valid = self.data.y_train[train_index], self.data.y_train[valid_index]
                model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nb_epoch,
                          validation_data=(x_valid, y_valid), callbacks=[model_checkpoint, early_stopping])
                print("model train finish: ", bmp, "at time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                valid_prd, test_prd = self.model_predict(model, val_data=(x_valid, y_valid))
                print("valid yun metric:", yun_metric(y_valid, valid_prd))
                if self.is_retrain:
                    valid_prd, test_prd = self.retrain(bmp, train_data=(x_train, y_train), valid_data=(x_valid, y_valid))
                test_prd_mean.append(test_prd)

            test_prd_mean = np.array(test_prd_mean)
            test_prd_mean = np.mean(test_prd_mean, axis=0)
            self._save_to_csv(self.data.test_id, test_prd_mean, bst_model_path)
        else:
            model = self.get_model()
            model.summary()
            x_train, x_valid, y_train, y_valid = train_test_split(self.data.x_train, self.data.y_train,
                                                                  random_state=2017,
                                                                  test_size=cfg.MODEL_FIT_validation_split)
            model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.nb_epoch,
                      validation_data=(x_valid, y_valid), callbacks=[model_checkpoint, early_stopping])
            print("model train finish: ", bst_model_path, "at time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            valid_prd, test_prd = self.model_predict(model, val_data=(x_valid, y_valid))
            print("valid yun metric:", yun_metric(y_valid, valid_prd))
            if self.is_retrain:
                valid_prd, test_prd = self.retrain(bst_model_path, train_data=(x_train, y_train),
                                                   valid_data=(x_valid, y_valid))
            self._save_to_csv(self.data.test_id, test_prd, bst_model_path)

    def retrain(self, bst_model_path, train_data, valid_data):
        """
        :param bst_model_path:
        :param train_data: (x_train, y_train)
        :param valid_data: (x_valid, y_valid)
        :return:
        """
        print('----> retrain model:', bst_model_path)
        model = self.get_model(trainable=True)
        model.load_weights(bst_model_path)
        new_path = bst_model_path + "_retrain"
        model_checkpoint = ModelCheckpoint(new_path, save_best_only=True, save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        call_back = [model_checkpoint, early_stopping]
        model.fit(train_data[0], train_data[1], batch_size=self.batch_size, epochs=self.nb_epoch,
                  validation_data=valid_data, callbacks=call_back)
        print("model retrain finish: ", new_path, "at time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        valid_prd, test_prd = self.model_predict(model, val_data=valid_data)
        print("model retrain,valid yun metric:", yun_metric(valid_data[1], valid_prd))
        return valid_prd, test_prd

    def model_predict(self, model, val_data):
        """
        :param model: keras model which has been trained
        :param val_data: 线下验证数据集(x_valid, y_valid)
        """
        valid_prd = model.predict(val_data[0])
        valid_prd = np.reshape(valid_prd, (valid_prd.shape[0],))
        test_prd = model.predict(self.data.x_test)
        test_prd = np.reshape(test_prd, (test_prd.shape[0],))
        return valid_prd, test_prd

    def _save_to_csv(self, ids, scores, path):
        assert len(ids) == len(scores)
        print(scores)
        sample_submission = pd.DataFrame({
            "Id": ids,
            "Score": scores
        })
        result_path = self._get_result_path(path)
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
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim
