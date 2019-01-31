import logging
import numpy as np
from abc import ABCMeta, abstractmethod
from keras_text_cls.vocab import Vocabulary, SYMBOL_PADDING, SYMBOL_UNKNOWN


class BaseEmbedder(object):
    __metaclass__ = ABCMeta

    def __init__(self, dim=300, seed=None):
        self._model = None
        self._dim = dim
        if seed is not None:
            np.random.seed(seed)
        self._UNKNOWN_VEC = np.random.rand(dim)
        self._PADDING_VEC = np.zeros(dim)
        self.is_fitted = False
        self._predefined_vocabs = [SYMBOL_PADDING, SYMBOL_UNKNOWN]

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
        if not self.is_fitted:
            raise ValueError("model needs to be fitted first")
        res = []
        for sentence in words:
            s = []
            for w in sentence:
                s.append(self.transform(w))
            res.append(s)
        return res

    def get_dim(self):
        """
        get the dim of the fitted embedder
        :return: dim
        """
        return self._dim

    @abstractmethod
    def fit_on_words(self, words, **kwargs):
        """
        fit on words
        :param words: 2-dim list of words
        :return: None
        """
        pass

    def build_embedding_matrix(self):
        """
        build embedding matrix and vocabulary
        :return: embedding matrix, vocabulary
        """
        if not self.is_fitted:
            raise ValueError("model needs to be fitted first")
        em_matrix = list()
        for w in self.get_vocabs():
            v = self.transform(w)
            em_matrix.append(v)
        em_matrix = np.array(em_matrix)
        return em_matrix

    @abstractmethod
    def get_vocabs(self):
        """
        get vocabs
        :return: list of words, with pre-defined tokens
        """
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @staticmethod
    @abstractmethod
    def load_model(path):
        pass

