# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from data_process import MAX_FEATURE
from base_model import TextModel
from metric import tensor_yun_loss


class AttentionLSTM(TextModel):
    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = Embedding(MAX_FEATURE, self.embed_size, input_length=self.max_len)(inputs)
        x = Bidirectional(LSTM(128, return_sequences=True))(emb)
        x = self.attention_3d_block(x)
        x = GlobalMaxPool1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(6, activation=self.last_act)(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}.h5".format(pre=self.__class__.__name__, act=self.last_act,
                                                               epo=self.nb_epoch,
                                                               embed=self.embed_size, max_len=self.max_len)

    def attention_3d_block(self, inputs):
        """
        attention mechanisms for lstm
        :param inputs: shape (batch_size, seq_length, input_dim)
        :return:
        """
        a = Permute((2, 1))(inputs)
        a = Dense(self.max_len, activation='softmax')(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        att_mul = multiply([inputs, a_probs])
        return att_mul
