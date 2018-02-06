# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from base_model import TextModel
from data_process import cfg, get_embedding_layer
from metric import tensor_yun_loss


class TextRCNN(TextModel):

    def __init__(self, filter_window=3, **kwargs):
        self.filter_window = filter_window
        super(TextRCNN, self).__init__(**kwargs)

    def get_model(self):
        inputs = Input(shape=(self.max_len,))
        emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                  use_pretrained=self.use_pretrained, trainable=self.trainable)(inputs)
        x = Bidirectional(LSTM(cfg.LSTM_hidden_size, return_sequences=True))(emb)
        x = concatenate([x, emb], axis=2)

        x = Conv1D(128, self.filter_window, activation='tanh')(x)
        x = GlobalMaxPool1D()(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.num_class, activation=self.last_act)(x)  # softmax
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{wind}_{mwl}_{time}_{inp_num}_upt-{upt}_tn-{tn}_ser-{ser}_cls-{cls}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch,
            embed=self.embed_size, max_len=self.max_len, wind=self.filter_window,
            time=self.time, mwl=self.min_word_len, inp_num=self.inputs_num, cls=self.num_class,
            upt=int(self.use_pretrained), tn=int(self.trainable), ser=int(self.data.serial)
        )
