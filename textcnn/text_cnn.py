# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from base_model import TextModel
from data_process import get_embedding_layer


class TextCNN(TextModel):

    def __init__(self, filter_window=3, **kwargs):
        self.filter_window = filter_window
        super(TextCNN, self).__init__(**kwargs)

    def get_model(self, trainable=None):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=trainable,
                                  use_new_vector=self.use_new_vector)(inputs)
        x = Conv1D(128, self.filter_window, activation='relu')(emb)
        x = GlobalMaxPool1D()(x)

        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch, embed=self.embed_size, max_len=self.max_len,
            wind=self.filter_window, time=self.time,
            upt=int(self.use_pretrained), tn=int(self.trainable))


class TextCNNBN(TextModel):

    def __init__(self, filter_window=3, **kwargs):
        self.filter_window = filter_window
        super(TextCNNBN, self).__init__(**kwargs)

    def get_model(self, trainable=None):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=trainable,
                                  use_new_vector=self.use_new_vector)(inputs)
        x = Conv1D(128, self.filter_window)(emb)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalMaxPool1D()(x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch, embed=self.embed_size, max_len=self.max_len,
            wind=self.filter_window, time=self.time,
            upt=int(self.use_pretrained), tn=int(self.trainable))
