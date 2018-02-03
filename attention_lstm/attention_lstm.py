# -*- coding: utf-8 -*-

import sys
from keras.layers import *
from keras.models import Model

sys.path.append("../")

from data_process import get_embedding_layer
from base_model import TextModel
from metric import tensor_yun_loss


class AttentionLSTM(TextModel):
    def get_model(self):
        inputs, x = self._get_multi_input(self.inputs_num)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(5, activation=self.last_act)(x)
        model = Model(inputs=inputs, outputs=x)
        model.compile(loss='mse', optimizer=self.optimizer, metrics=['acc', 'mse', tensor_yun_loss])
        return model

    def _get_bst_model_path(self):
        return "{pre}_{act}_{epo}_{embed}_{max_len}_{mwl}_{time}_{inp_num}_upt-{upt}_tn-{tn}_ser-{ser}.h5".format(
            pre=self.__class__.__name__, act=self.last_act, epo=self.nb_epoch, inp_num=self.inputs_num,
            embed=self.embed_size, max_len=self.max_len, time=self.time, mwl=self.min_word_len,
            upt=int(self.use_pretrained), tn=int(self.trainable), ser=int(self.data.serial)
        )

    def _get_multi_input(self, num):
        inputs = []
        outputs = []
        for _ in range(num):
            inp = Input(shape=(self.max_len,))
            emb = get_embedding_layer(self.data.tokenizer, max_len=self.max_len, embedding_dim=self.embed_size,
                                      use_pretrained=self.use_pretrained, trainable=self.trainable)(inp)
            x = Bidirectional(LSTM(128, return_sequences=True))(emb)
            x = self.attention_3d_block(x)
            x = GlobalMaxPool1D()(x)
            outputs.append(x)
            inputs.append(inp)
        output = concatenate(outputs) if num >= 2 else outputs[0]
        return inputs, output

    def attention_3d_block(self, inputs):
        """
        attention mechanisms for lstm
        :param inputs: shape (batch_size, seq_length, input_dim)
        :return:
        """
        a = Permute((2, 1))(inputs)
        a = Dense(self.max_len, activation='softmax')(a)
        a_probs = Permute((2, 1))(a)    # attention_vec
        att_mul = multiply([inputs, a_probs])
        return att_mul
