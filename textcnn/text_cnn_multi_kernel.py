# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from base_model import TextModel
from data_process import get_embedding_layer
from metric import tensor_yun_loss


class TextCNNMultiKernel(TextModel):

    def __init__(self, **kwargs):
        self.filters = [2, 3, 4]
        super(TextCNNMultiKernel, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)

        concat_x = []
        for filter_size in self.filters:
            x = Conv1D(128, filter_size, activation='relu')(emb)
            x = GlobalMaxPool1D()(x)
            concat_x.append(x)

        x = concatenate(concat_x)

        x = Dropout(0.3)(x)
        x = Dense(5, activation=self.last_act)(x)  # softmax
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{wind}_{mwl}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, mwl=self.min_word_len, upt=int(self.use_pretrained), tn=int(self.trainable)
        )


class TextCNNMultiKernelBN(TextModel):

    def __init__(self, **kwargs):
        self.filters = [2, 3, 4]
        super(TextCNNMultiKernelBN, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)

        concat_x = []
        for filter_size in self.filters:
            x = Conv1D(128, filter_size)(emb)
            x = BatchNormalization()(x)
            x = GlobalMaxPool1D()(x)
            concat_x.append(x)

        x = concatenate(concat_x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(5, activation=self.last_act)(x)  # softmax
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{wind}_{mwl}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, mwl=self.min_word_len, upt=int(self.use_pretrained), tn=int(self.trainable)
        )
