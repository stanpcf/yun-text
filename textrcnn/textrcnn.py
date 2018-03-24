# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from base_model import TextModel
from data_process import cfg, get_embedding_layer


class TextRCNN(TextModel):

    def __init__(self, filter_window=3, **kwargs):
        self.filter_window = filter_window
        super(TextRCNN, self).__init__(**kwargs)

    def get_model(self, trainable=None):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)
        x = Bidirectional(LSTM(cfg.LSTM_hidden_size, return_sequences=True))(emb)
        x = concatenate([x, emb], axis=2)
        reshape = Reshape((self.max_len, self.embed_size + cfg.LSTM_hidden_size * 2, 1))(x)

        concat_x = []
        for filter_size in cfg.TEXT_CNN_filters:
            x = reshape
            x = self._conv_relu_maxpool(x, filter_size)
            concat_x.append(x)

        x = concatenate(concat_x, axis=1)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def _conv_relu_maxpool(self, inp, filter_size):
        x = Conv2D(128, kernel_size=(filter_size, self.embed_size + cfg.LSTM_hidden_size*2), activation='relu')(inp)
        x = MaxPool2D(pool_size=(self.max_len - filter_size + 1, 1), strides=(1, 1))(x)
        return x

    def _get_bst_model_path(self):
        return "{pre}_{epo}_{embed}_{max_len}_{wind}_{time}_upt-{upt}_tn-{tn}.h5".format(
            pre=self.__class__.__name__, epo=self.nb_epoch, embed=self.embed_size, max_len=self.max_len,
            wind=self.filter_window, time=self.time, upt=int(self.use_pretrained), tn=int(self.trainable))
