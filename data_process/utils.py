# coding: utf8

import os


def get_stop_words():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords.txt')
    with open(path) as f:
        words = [line.strip() for line in f]
    return words
