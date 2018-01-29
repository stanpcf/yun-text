#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model
sys.path.append("../")

from data_process import MAX_FEATURE
from base_model import TextModel
from metric import tensor_yun_loss


class BiLSTM(TextModel):
    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = Embedding(MAX_FEATURE, self.embed_size, input_length=self.max_len)(inputs)
        x = Bidirectional(LSTM(50, return_sequences=True))(emb)
        x = GlobalMaxPool1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = Dense(6, activation=self.last_act)(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}.h5".format(pre=self.__class__.__name__, act=self.last_act,
                                                               epo=self.nb_epoch,
                                                               embed=self.embed_size, max_len=self.max_len)
