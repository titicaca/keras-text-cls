import unittest
from keras_text_cls.vocab import Vocabulary


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        pass

    def test_vocab(self):
        words = [["我", "爱", "北京", "天安门"], ["天安门", "上", "太阳", "升"]]
        vocabulary = Vocabulary()
        vocabulary.fit_on_words(words)
        assert (vocabulary.vocab_size == 9)
        indices = vocabulary.words_to_idx(words)
        assert (len(indices) == 2)
        assert (indices[0] == [2, 3, 4, 5])
        assert (indices[1] == [5, 6, 7, 8])
        vocabulary.save_vocab("/tmp/tmp_vocab")
        vocab_loaded = Vocabulary.load_vocab("/tmp/tmp_vocab")
        assert (vocab_loaded.vocab_size == 9)
        assert (vocab_loaded.words_to_idx(words) == indices)
