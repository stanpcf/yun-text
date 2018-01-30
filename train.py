#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from tensorflow import flags

from data_process import get_data


flags.DEFINE_string('classifier', None, "path of the Class for the classifier")
flags.DEFINE_integer('nb_epoch', 50, "number of epoch")
flags.DEFINE_integer('max_len', 80, "regular sentence to a fixed length")
flags.DEFINE_integer('embed_size', 300, "hidden size of embedding layer")
flags.DEFINE_string('last_act', 'softmax', "the activation for the last layer")
flags.DEFINE_integer('batch_size', 640, "batch size for train")
flags.DEFINE_string('optimizer', 'adam', "the optimizer for train")
flags.DEFINE_bool('global_data', True, "if use global data for all classifier")
flags.DEFINE_float('train_size', 0.8, "the rate of train-valid split for train data set")
flags.DEFINE_bool('use_pretrained', False, "if use pretrained vector for embedding layer")
flags.DEFINE_bool('trainable', True,
                  "if the embedding layer is trainable. this param is used only `use_pretrained` is true")
flags.DEFINE_integer("conv_kernel", 3, "kernel size of TextCNN")

FLAGS = flags.FLAGS

register_model = [
    'attention_lstm.AttentionLSTM',
    'bidirectional_lstm.BiLSTM',
    'textcnn.TextCNN',
]


def main():
    max_len = FLAGS.max_len
    (x_train, y_train, sample_weights), (x_valid, y_valid, valid_id), x_test, test_id = \
        get_data(train_size=FLAGS.train_size, max_len=max_len, set_cls_weight=True)
    kwargs = {
        "x_train": x_train,
        "y_train": y_train,
        "sample_weights": sample_weights,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "valid_id": valid_id,
        "x_test": x_test,
        "test_id": test_id,
    }
    models_cls = [FLAGS.classifier] if FLAGS.classifier else register_model
    for cls_name in models_cls:
        module_name = ".".join(cls_name.split('.')[:-1])
        cls_name = cls_name.split('.')[-1]
        _module = importlib.import_module(module_name)
        cls = _module.__dict__.get(cls_name)
        model = cls(nb_epoch=FLAGS.nb_epoch, max_len=FLAGS.max_len, embed_size=FLAGS.embed_size,
                    last_act=FLAGS.last_act, batch_size=FLAGS.batch_size, optimizer=FLAGS.optimizer,
                    global_data=FLAGS.global_data, filter_window=FLAGS.conv_kernel, **kwargs)

        model.train()
        model.predict()


if __name__ == '__main__':
    main()
