# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from base_model import TextModel
from data_process import MAX_FEATURE
from metric import tensor_yun_loss


class TextCNN(TextModel):

    def __init__(self, filter_window=3, **kwargs):
        self.filter_window = filter_window
        super(TextCNN, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = Embedding(MAX_FEATURE, self.embed_size, input_length=self.max_len)(inputs)

        x = Conv1D(128, self.filter_window, activation='relu')(emb)
        x = GlobalMaxPool1D()(x)

        x = Dropout(0.3)(x)
        x = Dense(5, activation=self.last_act)(x)  # softmax
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{wind}_{mwl}_{time}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind=self.filter_window,
            time=self.time, mwl=self.min_word_len
        )


class TextCNNBN(TextModel):

    def __init__(self, filter_window=3, **kwargs):
        self.filter_window = filter_window
        super(TextCNNBN, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = Embedding(MAX_FEATURE, self.embed_size, input_length=self.max_len)(inputs)

        x = Conv1D(128, self.filter_window)(emb)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = GlobalMaxPool1D()(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(5, activation=self.last_act)(x)  # softmax
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{wind}_{time}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind=self.filter_window,
            time=self.time
        )
