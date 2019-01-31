import jieba
import logging
from keras_text_cls.dict import JIEBA_DICT_PATH, load_stop_words
from keras_text_cls.tokenizer.base_tokenizer import BaseTokenizer


class JiebaTokenizer(BaseTokenizer):
    """
    A Chinese Tokenizer Wrapper of Jieba
    Reference to: https://github.com/fxsjy/jieba
    """
    def __init__(self, **kwargs):
        jieba.load_userdict(JIEBA_DICT_PATH)
        logging.info("loaded jieba dict: " + JIEBA_DICT_PATH)
        self.stop_words = load_stop_words()

    def tokenize(self, body, enable_stop_words=False):
        if body is None:
            body = ''
        assert isinstance(body, str)
        words = jieba.cut(body)
        if enable_stop_words:
            words = (w for w in words if w not in self.stop_words)
        return words

