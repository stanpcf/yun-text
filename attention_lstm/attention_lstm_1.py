# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from data_process import get_embedding_layer
from base_model import TextModel, Attention


class AttentionLSTM1(TextModel):
    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)
        x = Bidirectional(LSTM(128, return_sequences=True))(emb)
        x = Attention(self.max_len)(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse'])
        return model

    """
    def get_model(self):
        inputs = Input(shape=(self.max_len,), dtype='int32')
        emb = get_embedding_layer(self.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)
        x = Bidirectional(LSTM(50, return_sequences=True))(emb)
        x = Dropout(0.25)(x)
        x = Attention(self.max_len)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization()(x)
        x = Dense(6, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['acc'])    # optimizer = rmsprop
        return model
    """

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch, embed=self.embed_size, max_len=self.max_len,
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))
