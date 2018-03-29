#!/usr/bin/env bash

python train.py --classifier=textcnn.TextCNNMultiKernel1D --embed_size=200 --notrainable --is_retrain --max_len=150
python train.py --classifier=textcnn.TextCNNMultiKernel1D1 --embed_size=200 --notrainable --is_retrain --max_len=150
