import logging
from keras_text_cls.tokenizer.base_tokenizer import BaseTokenizer
from keras_text_cls.dict import load_stop_words


class CharTokenizer(BaseTokenizer):
    """
    A character tokenizer
    """
    def __init__(self, **kwargs):
        self.stop_words = load_stop_words()

    def tokenize(self, body, enable_stop_words=False):
        if body is None:
            body = ''
        assert isinstance(body, str)
        words = list(body)
        if enable_stop_words:
            words = (w for w in words if w not in self.stop_words)
        return words

