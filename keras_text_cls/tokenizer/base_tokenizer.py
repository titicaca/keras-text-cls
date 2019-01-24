import logging
from abc import ABCMeta, abstractmethod


class BaseTokenizer(object):
    def __init__(self, **kwargs):
        __metaclass__ = ABCMeta

    @abstractmethod
    def tokenize(self, body, **kwargs):
        # Split by " ", only used for english corpus
        return body.split(" ")


