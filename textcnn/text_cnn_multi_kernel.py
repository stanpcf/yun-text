# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from base_model import TextModel
from data_process import cfg, get_embedding_layer


class TextCNNMultiKernel(TextModel):

    def __init__(self, **kwargs):
        self.filters = cfg.TEXT_CNN_filters
        super(TextCNNMultiKernel, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)
        reshape = Reshape((self.max_len, self.embed_size, 1))(emb)

        concat_x = []
        for filter_size in self.filters:
            x = reshape
            x = self._conv_relu_maxpool(x, filter_size)
            concat_x.append(x)

        x = concatenate(concat_x, axis=1)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(8, activation='relu')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _conv_relu_maxpool(self, inp, filter_size):
        x = Conv2D(64, kernel_size=(filter_size, self.embed_size), activation='relu')(inp)
        x = MaxPool2D(pool_size=(self.max_len - filter_size + 1, 1), strides=(1, 1))(x)
        # x = AvgPool2D(pool_size=(self.max_len - filter_size + 1, 1), strides=(1, 1))(x)
        return x

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))


class TextCNNMultiKernelBN(TextModel):

    def __init__(self, **kwargs):
        self.filters = cfg.TEXT_CNN_filters
        super(TextCNNMultiKernelBN, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)
        reshape = Reshape((self.max_len, self.embed_size, 1))(emb)

        concat_x = []
        for filter_size in self.filters:
            x = reshape
            x = self._conv_bn_relu_maxpool(x, filter_size)
            concat_x.append(x)

        x = concatenate(concat_x, axis=1)
        x = Flatten()(x)
        x = Dense(8, activation='tanh')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _conv_bn_relu_maxpool(self, inp, filter_size):
        x = Conv2D(64, kernel_size=(filter_size, self.embed_size), activation='linear')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D(pool_size=(self.max_len - filter_size + 1, 1), strides=(1, 1))(x)
        return x

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))


class TextCNNMultiKernel1D(TextModel):

    def __init__(self, **kwargs):
        self.filters = cfg.TEXT_CNN_filters
        super(TextCNNMultiKernel1D, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)

        concat_x = []
        for filter_size in self.filters:
            x = emb
            x = self._conv_relu_maxpool(x, filter_size)
            concat_x.append(x)

        x = concatenate(concat_x, axis=1)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _conv_relu_maxpool(self, inp, filter_size):
        x = Conv1D(128, kernel_size=filter_size, activation='relu')(inp)
        x = GlobalMaxPool1D()(x)
        return x

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))


class TextCNNMultiKernel1D2CNN(TextModel):

    def __init__(self, **kwargs):
        self.filters = cfg.TEXT_CNN_filters
        super(TextCNNMultiKernel1D2CNN, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)

        concat_x = []
        for filter_size in self.filters:
            x = emb
            x = Conv1D(128, kernel_size=filter_size, activation='relu')(x)
            x = Conv1D(64, kernel_size=filter_size, activation='relu')(x)
            x = GlobalMaxPool1D()(x)
            concat_x.append(x)

        x = concatenate(concat_x, axis=1)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))


class TextCNNMultiKernelBN1D(TextModel):

    def __init__(self, **kwargs):
        self.filters = cfg.TEXT_CNN_filters
        super(TextCNNMultiKernelBN1D, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)

        concat_x = []
        for filter_size in self.filters:
            x = emb
            x = self._conv_bn_relu_maxpool(x, filter_size)
            concat_x.append(x)

        x = concatenate(concat_x, axis=1)
        x = Flatten()(x)
        x = Dense(8, activation='tanh')(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _conv_bn_relu_maxpool(self, inp, filter_size):
        x = Conv1D(128, kernel_size=filter_size, activation='relu')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = GlobalMaxPool1D()(x)
        return x

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind="-".join([str(s) for s in self.filters]),
            time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))
