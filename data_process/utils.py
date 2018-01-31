# coding: utf8

from pathlib import Path


def get_stop_words():
    path = Path(__file__).absolute().parent / 'stopwords.txt'
    with open(path) as f:
        words = [line.strip() for line in f]
    return frozenset(words)
