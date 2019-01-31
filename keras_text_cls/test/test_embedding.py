import unittest
from keras_text_cls.embedding import *


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        pass

    def test_word2vec(self):
        w2v = Word2vecEmbedder(dim=128)
        assert not w2v.is_fitted
        words = [["我", "爱", "北京", "天安门"], ["天安门", "上", "太阳", "升"]]
        w2v.fit_on_words(words, min_count=1)
        assert w2v.is_fitted
        vocabulary = Vocabulary(w2v.get_vocabs())
        em_matrix = w2v.build_embedding_matrix()
        assert em_matrix.shape[1] == 128
        assert len(em_matrix) == vocabulary.vocab_size
        vectors = w2v.transform_on_words(words)
        assert len(vectors) == 2
        assert len(vectors[0]) == 4
        assert len(vectors[0][0]) == 128
        v1 = w2v.transform("X")
        assert (v1 == w2v._UNKNOWN_VEC).all()
        v2 = w2v.transform(SYMBOL_PADDING)
        assert (v2 == w2v._PADDING_VEC).all()
        # test saving
        with tempfile.NamedTemporaryFile(suffix="_w2v.model", mode='w') as tmp:
            path = tmp.name
        w2v.save_model(path)
        w2v_loaded = Word2vecEmbedder.load_model(path)
        assert w2v.get_dim() == w2v_loaded.get_dim()
        os.remove(path)

    def test_fasttext(self):
        ft = FasttextEmbedder(dim=128)
        assert not ft.is_fitted
        words = [["我", "爱", "北京", "天安门"], ["天安门", "上", "太阳", "升"]]
        ft.fit_on_words(words, temp_file="/tmp/tmp_fasttext.txt", model_path="/tmp/tmp_fasttext.model",
                        thread=1, sg=1, min_count=1)
        assert ft.is_fitted
        vocabulary = Vocabulary(ft.get_vocabs())
        em_matrix = ft.build_embedding_matrix()
        assert em_matrix.shape[1] == 128
        assert len(em_matrix) == vocabulary.vocab_size
        vectors = ft.transform_on_words(words)
        assert len(vectors) == 2
        assert len(vectors[0]) == 4
        assert len(vectors[0][0]) == 128
        v1 = ft.transform("X")
        assert (v1 == ft._UNKNOWN_VEC).all()
        v2 = ft.transform(SYMBOL_PADDING)
        assert (v2 == ft._PADDING_VEC).all()
