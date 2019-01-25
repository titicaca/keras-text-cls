import unittest
from keras_text_cls.model import *
from keras_text_cls.tokenizer import *
from keras_text_cls.vocab import *


class TestModel(unittest.TestCase):
    def setUp(self):
        pass

    def test_text_mlp(self):
        jieba_tokenizer = JiebaTokenizer()
        text = ['我爱北京天安门', '天安门上太阳升']
        labels = np.array([0, 1])
        words = list(jieba_tokenizer.text_to_words(text))
        vocabulary = Vocabulary()
        vocabulary.fit_on_words(words)
        indices = vocabulary.words_to_idx(words)
        max_seq = 10
        text_mlp = TextMLP(num_classes=2, vocabs=vocabulary.vocabs, num_hidden_units=[10], max_seq_len=max_seq,
                           embedding_dim=10)
        text_mlp.compile(loss='categorical_crossentropy',
                optimizer='adam',
              metrics=['acc'])
        inputs = pad_sequences(indices, maxlen=max_seq, padding='post', value=0.)
        text_mlp.fit(inputs, labels, epochs=100, batch_size=10)
        preds = text_mlp.predict(inputs)
        assert (len(preds) > 0)
