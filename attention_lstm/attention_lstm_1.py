# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from data_process import MAX_FEATURE
from base_model import TextModel, Attention
from metric import tensor_yun_loss


class AttentionLSTM1(TextModel):
    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = Embedding(MAX_FEATURE, self.embed_size, input_length=self.max_len)(inputs)
        x = Bidirectional(LSTM(256, return_sequences=True))(emb)
        x = Attention(self.max_len)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(6, activation=self.last_act)(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
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
        return "{pre}_{act}_{epo}_{embed}_{max_len}.h5".format(pre=self.__class__.__name__, act=self.last_act,
                                                               epo=self.nb_epoch,
                                                               embed=self.embed_size, max_len=self.max_len)
