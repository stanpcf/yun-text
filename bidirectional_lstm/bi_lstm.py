#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model
sys.path.append("../")

from data_process import cfg, get_embedding_layer
from base_model import TextModel
from metric import tensor_yun_loss


class BiLSTM(TextModel):
    """该模型使用多种分词工具之后的切割作为输入"""
    def get_model(self):
        inputs, x = self._get_multi_input(self.inputs_num)

        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(5, activation=self.last_act)(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_multi_input(self, num):
        inputs = []
        outputs = []
        for _ in range(num):
            inp = Input(shape=(self.max_len,))
            emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                      use_pretrained=self.use_pretrained, trainable=self.trainable)(inp)
            x = Bidirectional(LSTM(cfg.LSTM_hidden_size, return_sequences=True))(emb)
            x = GlobalMaxPool1D()(x)
            outputs.append(x)
            inputs.append(inp)
        output = concatenate(outputs) if num >= 2 else outputs[0]
        return inputs, output

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{mwl}_{time}_{inp_num}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch, inp_num=self.inputs_num,
            embed=self.embed_size, max_len=self.max_len, time=self.time, mwl=self.min_word_len,
            upt=int(self.use_pretrained), tn=int(self.trainable)
        )
