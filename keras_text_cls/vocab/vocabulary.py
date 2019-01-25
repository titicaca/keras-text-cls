import logging
from keras_text_cls.vocab.util import *


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNKNOWN>": 1}
        self.vocabs = ["<PAD>", "<UNKNOWN>"]
        self.num_words = 0
        self.word_freq = {}
        # TODO add TF-IDF calculation

    def fit_on_words(self, text_words):
        """
        Initialize a vocabulary via fitting on a list of text tokens
        :param text_words: list of text words
        :return: None
        """
        idx = 1
        for sentence in text_words:
            if not isinstance(sentence, list):
                tokens = [sentence]
            else:
                tokens = flatten_list_dfs(sentence)
            for w in tokens:
                if w not in self.word2idx.keys():
                    idx += 1
                    self.word2idx[w] = idx
                    self.vocabs.append(w)
                    assert(self.vocabs[idx] == w)
                    self.num_words += 1
                    self.word_freq[w] = 1
                else:
                    self.word_freq[w] += 1

    def words_to_idx(self, words, dim=2):
        """
        transform words to indices
        :param words: list of words
        :param dim: indicating the dim of words, either 1 or 2, default is 2
        :return: list of indices
        """
        assert(dim == 1 or dim == 2)
        if self.num_words == 0:
            raise ValueError("dict is empty, fit_on_words need to be called first to initialize the vocabulary")
        if dim == 1:
            return [self.word2idx[w] if w in self.word2idx.keys() else 1 for w in words]
        else:
            return [[self.word2idx[w] if w in self.word2idx.keys() else 1
                     for w in sentence_tokens] for sentence_tokens in words]

    def idx_to_words(self, indices, dim=2):
        """
        transform indices to words
        :param indices: list of words
        :param dim: indicating the dim of indices, either 1 or 2, default is 2
        :return: list of words
        """
        assert(dim == 1 or dim == 2)
        if self.num_words == 0:
            raise ValueError("dict is empty, fit_on_words need to be called first to initialize the vocabulary")
        if dim == 1:
            return [self.vocabs[idx] for idx in indices]
        else:
            return [[self.idx_to_words[idx] for idx in sentence_indices] for sentence_indices in indices]
