import logging
from abc import ABCMeta, abstractmethod


class BaseTokenizer(object):
    def __init__(self, **kwargs):
        __metaclass__ = ABCMeta

    def tokenize(self, body, **kwargs):
        # Split by " ", only used for english corpus
        return body.split(" ")

    def text_to_words(self, text):
        """
        transform text to list of words
        :param text: text is a list of sentences or documents
        :return: a 2d list of word tokens
        """
        res = []
        for sentence in text:
            res.append(list(self.tokenize(sentence)))
        return res
