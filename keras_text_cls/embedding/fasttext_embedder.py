import logging
from keras_text_cls.embedding.base_embedder import BaseEmbedder
from keras_text_cls.vocab import Vocabulary, SYMBOL_PADDING, SYMBOL_UNKNOWN
import numpy as np
import fasttext
import tempfile
import os


class FasttextEmbedder(BaseEmbedder):
    """
    FasttextEmbedder is a wrapper of fasttext
    Reference to: https://pypi.org/project/fasttext/

    Attributes
    ----------
    dim: int
        embedding vector dimension, default 300
    seed: int
        random seed, default is None
    """
    def __init__(self, dim=300, seed=None):
        super().__init__(dim, seed)

    def transform(self, word):
        if not self.is_fitted:
            raise ValueError("model needs to be fitted first")
        if word in self._model.words:
            return self._model[word]
        elif word == SYMBOL_PADDING:
            return self._PADDING_VEC
        else:
            return self._UNKNOWN_VEC

    def fit_on_words(self, words, temp_file=None, sg=0, model_path=None, **kwargs):
        """
        fit fasttext model on words, vector size is assigned internally by default (equal to model._dim)
        parameters are the same as fasttext model
        :param words: 2d list of str
            2d list of words
        :param temp_file: str
            temp file path for training the model
        :param model_path: str
            model path for trained fasttext model
        :param sg: int {1, 0}
            Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used
        :param kwargs: dict of params
            List of available params and their default value from fasttext (https://pypi.org/project/fasttext)
            lr             learning rate [0.05]
            lr_update_rate change the rate of updates for the learning rate [100]
            dim            size of word vectors [100]
            ws             size of the context window [5]
            epoch          number of epochs [5]
            min_count      minimal number of word occurences [5]
            neg            number of negatives sampled [5]
            word_ngrams    max length of word ngram [1]
            loss           loss function {ns, hs, softmax} [ns]
            bucket         number of buckets [2000000]
            minn           min length of char ngram [3]
            maxn           max length of char ngram [6]
            thread         number of threads [12]
            t              sampling threshold [0.0001]
            silent         disable the log output from the C++ extension [1]
            encoding       specify input_file encoding [utf-8]
        :return: fitted model
        """
        if temp_file is None:
            with tempfile.NamedTemporaryFile(suffix="_fasttext_words.txt") as tmp:
                temp_file = tmp.name
        if model_path is None:
            with tempfile.NamedTemporaryFile(suffix="_fasttext.model") as tmp:
                model_path = tmp.name

        with open(temp_file, 'w') as f:
            for i in range(len(words)):
                s = ' '.join(words[i]) + '\n'
                f.write(s)
            f.close()
        try:
            if sg == 1:
                self._model = fasttext.skipgram(input_file=temp_file, output=model_path, dim=self._dim, **kwargs)
            else:
                self._model = fasttext.cbow(input_file=temp_file, output=model_path, dim=self._dim, **kwargs)
        except Exception as ex:
            logging.error("error in fasttext model training")
            raise ex
        finally:
            os.remove(temp_file)
        logging.info("fasttext model is fitted successfully, model_path: " + model_path)
        self.is_fitted = True
        return self

    def get_vocabs(self):
        if not self.is_fitted:
            raise ValueError("model needs to be fitted first")
        vocabs_set = set(self._predefined_vocabs)
        vocabs = self._predefined_vocabs.copy()
        for w in self._model.words:
            if w not in vocabs_set:
                vocabs_set.add(w)
                vocabs.append(w)
        return vocabs

    def get_dim(self):
        if self.is_fitted:
            return self._model.dim
        else:
            return self._dim

    def save_model(self, path):
        raise ValueError("Fasttext model is already saved in fit_on_words method")

    @staticmethod
    def load_model(path):
        ft = FasttextEmbedder()
        ft._model = fasttext.load_model(path)
        logging.info("loaded fasttext model from " + path)
        ft.is_fitted = True
        return ft
