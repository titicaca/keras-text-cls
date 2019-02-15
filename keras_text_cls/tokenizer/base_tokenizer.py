import logging
from abc import ABCMeta, abstractmethod
from tqdm import tqdm


class BaseTokenizer(object):
    def __init__(self, **kwargs):
        __metaclass__ = ABCMeta

    def tokenize(self, body, **kwargs):
        # Split by " ", only used for english corpus
        return body.split(" ")

    def text_to_words(self, text, dim=1, **kwargs):
        """
        transform text to list of words
        :param text: text is a list of sentences or documents
        :return: list of word tokens
        """
        res = []
        assert dim == 1 or dim == 2
        if dim == 1:
            for doc in tqdm(text):
                res.append(list(self.tokenize(doc, **kwargs)))
            return res
        else:
            for doc in tqdm(text):
                res_s = []
                for sentence in doc:
                    res_s.append(list(self.tokenize(sentence, **kwargs)))
                res.append(res_s)
            return res
