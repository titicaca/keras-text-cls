import logging
from keras_text_cls.embedding.base_embedder import BaseEmbedder
import fasttext


class FasttextEmbedder(BaseEmbedder):

    def transform(self, words):
        pass

    def fit_on_words(self, words):
        pass

    def build_embedding_matrix_and_vocab(self):
        pass

    def save_model(self, path):
        pass

    @staticmethod
    def load_model(path):
        ft = FasttextEmbedder()
        ft._model = fasttext.load_model(path)
        logging.info("loaded fasttext model from " + path)
        return ft
