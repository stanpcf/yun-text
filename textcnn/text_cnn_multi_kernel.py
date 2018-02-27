# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from base_model import TextModel
from data_process import cfg, get_embedding_layer
from metric import tensor_yun_loss


class TextCNNMultiKernel(TextModel):
    """目前该model只支持一个inputs"""

    def __init__(self, **kwargs):
        self.filters = cfg.TEXT_CNN_filters
        super(TextCNNMultiKernel, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)

        concat_x = []
        for filter_size in self.filters:
            x = emb
            for _ in range(cfg.TEXT_CNN_CONV_NUM):
                x = self._conv_relu_maxpool(x, filter_size)
            concat_x.append(x)

        x = concatenate(concat_x)

        if self.one_hot:
            x = Dropout(0.5)(x)
            x = Dense(self.num_class, activation=self.last_act)(x)  # softmax
        else:
            x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _conv_relu_maxpool(self, inp, filter_size):
        x = Conv1D(128, filter_size, activation='relu')(inp)
        x = GlobalMaxPool1D()(x)
        return x

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}_ser-{ser}_{cn}_cls-{cls}_reg-{reg}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch, cls=self.num_class,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable),
            cn=cfg.TEXT_CNN_CONV_NUM, ser=int(self.data.serial), reg=int(not self.one_hot)
        )


class TextCNNMultiKernelBN(TextModel):
    """目前该model只支持一个inputs"""

    def __init__(self, **kwargs):
        self.filters = cfg.TEXT_CNN_filters
        super(TextCNNMultiKernelBN, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)

        concat_x = []
        assert cfg.TEXT_CNN_CONV_NUM > 0
        for filter_size in self.filters:
            x = emb
            for _ in range(cfg.TEXT_CNN_CONV_NUM):
                x = self._conv_bn_relu_maxpool(x, filter_size)
            concat_x.append(x)

        x = concatenate(concat_x)

        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if self.one_hot:
            x = Dense(self.num_class, activation=self.last_act)(x)  # softmax
        else:
            x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _conv_bn_relu_maxpool(self, inp, filter_size):
        x = Conv1D(128, filter_size)(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalMaxPool1D()(x)
        return x

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}_ser-{ser}_{cn}_cls-{cls}_reg-{reg}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch, cls=self.num_class,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable),
            cn=cfg.TEXT_CNN_CONV_NUM, ser=int(self.data.serial), reg=int(not self.one_hot)
        )
