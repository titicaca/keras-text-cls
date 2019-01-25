import unittest
from keras_text_cls.tokenizer import *


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        pass

    def test_jieba_tokenizer(self):
        jieba_tokenizer = JiebaTokenizer()
        text = ['我爱北京天安门', '天安门上太阳升']
        tokens = list(jieba_tokenizer.tokenize(text[0]))
        assert (len(tokens) == 4)

        words = jieba_tokenizer.text_to_words(text)
        assert (len(words) == 2)
        assert (words[0] == ["我", "爱", "北京", "天安门"])

    def test_char_tokeizer(self):
        char_tokenizer = CharTokenizer()
        text = ['我爱北京天安门', '天安门上太阳升']
        tokens = list(char_tokenizer.tokenize(text[0]))
        assert (len(tokens) == 7)

        words = char_tokenizer.text_to_words(text)
        assert (len(words) == 2)
        assert words[0] == ["我", "爱", "北", "京", "天", "安", "门"]
