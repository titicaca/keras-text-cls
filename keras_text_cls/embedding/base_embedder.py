import logging
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseEmbedder(object):
    __metaclass__ = ABCMeta

    def __init__(self, dim=300, seed=None):
        self._model = None
        self._dim = dim
        if seed is not None:
            np.random.seed(seed)
        self._UNKNOWN_VEC = np.random.rand(dim)
        self._PADDING_VEC = np.zeros(dim)

    @abstractmethod
    def transform(self, word):
        """
        transform a give word into the embedding vector
        :param word: word
        :return: embedding vector
        """
        pass

    def transform_on_words(self, words):
        """
        transform a list of words into embedded vectors
        :param words: 2-dim list of words
        :return: 3-dim list of vectors
        """
        res = []
        for sentence in words:
            s = []
            for w in sentence:
                s.append(self.transform(w))
            res.append(s)
        return res

    @abstractmethod
    def fit_on_words(self, words, **kwargs):
        """
        fit on words
        :param words: 2-dim list of words
        :return: None
        """
        pass

    @abstractmethod
    def build_embedding_matrix_and_vocab(self):
        """
        build embedding matrix and vocabulary
        :return: embedding matrix, vocabulary
        """
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @staticmethod
    @abstractmethod
    def load_model(path):
        pass

