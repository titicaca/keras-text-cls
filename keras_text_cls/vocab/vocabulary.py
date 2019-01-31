import logging
from keras_text_cls.vocab.util import *
import pickle
from tqdm import tqdm

SYMBOL_PADDING = "<PADDING>"
SYMBOL_UNKNOWN = "<UNKNOWN>"


class Vocabulary(object):
    """
    Vocabulary converting word to idx and idx to word

    Attributes
    ----------
    vocabs: str iterator
        vocabularies, words can not be duplicate in the vocab array
    """
    def __init__(self, vocabs=None):
        if vocabs is None:
            self.word2idx = {SYMBOL_PADDING: 0, SYMBOL_UNKNOWN: 1}
            self.idx2word = {0: SYMBOL_PADDING, 1: SYMBOL_UNKNOWN}
            self.vocab_size = 2
            self.word_freq = {}
            self.is_fitted = False
        else:
            # initialized by the pre-defined vocabs
            self.word2idx = {SYMBOL_PADDING: 0, SYMBOL_UNKNOWN: 1}
            self.idx2word = {0: SYMBOL_PADDING, 1: SYMBOL_UNKNOWN}
            self.vocab_size = 2
            self.word_freq = {}
            for w in vocabs:
                if w in self.word2idx.keys():
                    if w == SYMBOL_PADDING or w == SYMBOL_UNKNOWN:
                        continue
                    else:
                        raise ValueError("duplicate word found: " + w)
                idx = self.vocab_size
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                self.word_freq[w] = 1
                self.vocab_size += 1
            self.is_fitted = True

    def fit_on_words(self, text_words):
        """
        Initialize a vocabulary via fitting on a list of text tokens
        :param text_words: list of text words
        :return: None
        """
        if self.is_fitted:
            raise ValueError("vocabulary is already fitted")

        idx = 2
        for sentence in text_words:
            if not isinstance(sentence, list):
                tokens = [sentence]
            else:
                tokens = flatten_list_dfs(sentence)
            for w in tokens:
                if w not in self.word2idx.keys():
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
                    self.vocab_size += 1
                    self.word_freq[w] = 1
                    idx += 1
                else:
                    self.word_freq[w] += 1
        self.is_fitted = True
        return self

    def words_to_idx(self, words, dim=2):
        """
        transform words to indices
        :param words: list of words
        :param dim: indicating the dim of words, either 1 or 2, default is 2
        :return: list of indices
        """
        if not self.is_fitted:
            raise ValueError("vocabulary need to be fitted first")
        assert(dim == 1 or dim == 2)
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
        if not self.is_fitted:
            raise ValueError("vocabulary need to be fitted first")
        assert(dim == 1 or dim == 2)
        if dim == 1:
            return [self.idx2word[idx] for idx in indices]
        else:
            return [[self.idx_to_words[idx] for idx in sentence_indices] for sentence_indices in indices]

    def save_vocab(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_vocab(path):
        with open(path, 'rb') as file:
            vocab_pickle = pickle.load(file)
        return vocab_pickle
