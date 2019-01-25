import unittest
from keras_text_cls.vocab import Vocabulary


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        pass

    def test_vocab(self):
        words = [["我", "爱", "北京", "天安门"], ["天安门", "上", "太阳", "升"]]
        vocabulary = Vocabulary()
        vocabulary.fit_on_words(words)
        assert (vocabulary.num_words == 7)
        indices = vocabulary.words_to_idx(words)
        assert (len(indices) == 2)
        assert (indices[0] == [2, 3, 4, 5])
        assert (indices[1] == [5, 6, 7, 8])
