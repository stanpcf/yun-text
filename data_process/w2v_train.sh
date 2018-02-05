#!/usr/bin/env bash

python w2v.py --hidden_dim=100
python w2v.py --hidden_dim=100 --min_count=2
python w2v.py --hidden_dim=100 --window=3

python w2v.py --hidden_dim=200
python w2v.py --hidden_dim=200 --window=3

python w2v.py --hidden_dim=300
python w2v.py --hidden_dim=300 --window=3
