#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
from tensorflow import flags

from data_process import get_data

# model relation
flags.DEFINE_string('classifier', 'bidirectional_lstm.BiLSTM', "path of the Class for the classifier")
flags.DEFINE_integer('nb_epoch', 50, "number of epoch")
flags.DEFINE_integer('embed_size', 300, "hidden size of embedding layer")
flags.DEFINE_integer('batch_size', 640, "batch size for train")
flags.DEFINE_string('optimizer', 'adam', "the optimizer for train")
flags.DEFINE_bool('use_pretrained', True, "if use pretrained vector for embedding layer")
flags.DEFINE_bool('trainable', True,
                  "if the embedding layer is trainable. this param is used only `use_pretrained` is true")
# data relation
flags.DEFINE_float('train_size', 0.9, "the rate of train-valid split for train data set")
flags.DEFINE_integer('max_len', 80, "regular sentence to a fixed length")
flags.DEFINE_string("bst_model_path", None, "h5 path")
flags.DEFINE_boolean('is_kfold', False, "is kfold")
flags.DEFINE_integer('kfold', 8, "k when kfold is true")

FLAGS = flags.FLAGS


def main():
    data = get_data(train_size=FLAGS.train_size, max_len=FLAGS.max_len)

    cls_name = FLAGS.classifier
    module_name = ".".join(cls_name.split('.')[:-1])
    cls_name = cls_name.split('.')[-1]
    _module = importlib.import_module(module_name)
    cls = _module.__dict__.get(cls_name)

    model = cls(data=data, nb_epoch=FLAGS.nb_epoch, max_len=FLAGS.max_len, embed_size=FLAGS.embed_size,
                batch_size=FLAGS.batch_size, optimizer=FLAGS.optimizer,
                use_pretrained=FLAGS.use_pretrained, trainable=FLAGS.trainable,
                is_kfold=FLAGS.is_kfold, kfold=FLAGS.kfold)

    bmp = FLAGS.bst_model_path
    bmp_list = [bmp + "_" + str(i) for i in range(1, FLAGS.kfold)]
    model.bst_model_path_list = bmp_list
    model.predict(bst_model_path=FLAGS.bst_model_path)
    for bmp in bmp_list:
        model.retrain(bmp)
    model.predict(bst_model_path=FLAGS.bst_model_path + "_retrain")


if __name__ == '__main__':
    main()
