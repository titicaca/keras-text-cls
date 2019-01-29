import logging
from keras_text_cls.embedding.base_embedder import BaseEmbedder
from gensim.models.word2vec import Word2Vec


class Word2vecEmbedder(BaseEmbedder):
    def __init__(self, dim=300, seed=None):
        super().__init__(dim, seed)

    def transform(self, word):
        if word in self._model.wv.vocab:
            return self._model.wv[word]
        else:
            self._UNKNOWN_VEC

    def fit_on_words(self, words, sg=0, window=5, min_count=5, workers=4, iter=5, negative=5, **kwargs):
        """
        fit word2vec model on words, vector size is assigned internally by default (equal to model._dim)
        parameters are the same as gensim word2vec model
        :param words: 2-dim list of words
        :param sg: int {1, 0}
            Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used.
        :param window: int
            The maximum distance between the current and predicted word within a sentence.
        :param min_count: int
            The maximum distance between the current and predicted word within a sentence.
        :param workers: int
            Use these many worker threads to train the model (=faster training with multicore machines).
        :param iter: int
            Number of iterations (epochs) over the corpus.
        :param negative: int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        :param kwargs: more arguments can assigned by referring to gensim word2vec model
        :return: fitted model
        """
        sentences = words
        vector_size = self._dim
        word2vec_model = Word2Vec(sentences, size=vector_size, sg=sg, window=window, min_count=min_count,
                                  workers=workers, iter=iter, negative=negative, **kwargs)
        self._model = word2vec_model
        logging.info("word2ec model is fitted successfully")
        return self

    def build_embedding_matrix_and_vocab(self):
        pass

    def save_model(self, path):
        self._model.save(path)
        logging.info("saving model into: " + path)

    @staticmethod
    def load_model(path):
        w2v = Word2vecEmbedder()
        w2v._model = Word2Vec.load(path)
        logging.info("loaded word2vec model from: " + path)
        return w2v
