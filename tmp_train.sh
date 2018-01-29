#!/usr/bin/env bash

python train.py --classifier=bidirectional_lstm.BiLSTM --train_size=0.9

python train.py --classifier=textcnn.TextCNN --train_size=0.9
python train.py --classifier=textcnn.TextCNNBN --train_size=0.9
python train.py --classifier=textcnn.TextCNNMultiKernel --train_size=0.9
python train.py --classifier=textcnn.TextCNNMultiKernelBN --train_size=0.9


python train.py --classifier=attention_lstm.AttentionLSTM --train_size=0.9

python train.py --classifier=attention_lstm.AttentionLSTM1 --train_size=0.9

