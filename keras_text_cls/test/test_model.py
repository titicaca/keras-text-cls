import unittest
from keras_text_cls.model import *
from keras_text_cls.tokenizer import *
from keras_text_cls.vocab import *
from keras_text_cls.embedding import *


class TestModel(unittest.TestCase):
    def setUp(self):
        self.text = ['我爱北京天安门', '天安门上太阳升']

    def test_text_mlp(self):
        jieba_tokenizer = JiebaTokenizer()
        text = self.text
        labels = np.array([[0, 1], [1, 0]])
        words = jieba_tokenizer.text_to_words(text)
        vocabulary = Vocabulary()
        vocabulary.fit_on_words(words)
        indices = vocabulary.words_to_idx(words)
        max_seq = 10
        text_mlp = TextMLP(num_classes=2,
                           embedding_dim=128, embedding_trainable=True, embedding_vocab_size=vocabulary.vocab_size,
                           dropout=0.1,
                           num_hidden_units=[10], max_seq_len=max_seq, multi_label=False)
        text_mlp.compile(loss='categorical_crossentropy',
                optimizer='adam',
              metrics=['acc'])
        inputs = pad_sequences(indices, maxlen=max_seq, padding='post', value=0.)
        text_mlp.fit(inputs, labels, epochs=100, batch_size=10)
        y_prob = text_mlp.predict(inputs)
        y_pred = (y_prob >= 0.5).astype(int)
        assert (y_pred == labels).all()

    def test_text_mlp_with_trained_embedding_matrix(self):
        jieba_tokenizer = JiebaTokenizer()
        text = self.text
        labels = np.array([[0, 1], [1, 0]])
        words = jieba_tokenizer.text_to_words(text)
        w2v_embedding = Word2vecEmbedder(dim=128)
        w2v_embedding.fit_on_words(words, min_count=1)
        vocabulary = Vocabulary(w2v_embedding.get_vocabs())
        indices = vocabulary.words_to_idx(words)
        em_matrix = w2v_embedding.build_embedding_matrix()
        max_seq = 10
        text_mlp = TextMLP(num_classes=2,
                           embedding_dim=128, embedding_trainable=True, embedding_vocab_size=vocabulary.vocab_size,
                           embedding_matrix=em_matrix, dropout=0.1,
                           num_hidden_units=[10], max_seq_len=max_seq, multi_label=False)
        text_mlp.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
        inputs = pad_sequences(indices, maxlen=max_seq, padding='post', value=0.)
        text_mlp.fit(inputs, labels, epochs=100, batch_size=10)
        y_prob = text_mlp.predict(inputs)
        y_pred = (y_prob >= 0.5).astype(int)
        assert (y_pred == labels).all()

    def test_text_cnn(self):
        jieba_tokenizer = JiebaTokenizer()
        text = self.text
        labels = np.array([[0, 1], [1, 0]])
        words = jieba_tokenizer.text_to_words(text)
        vocabulary = Vocabulary()
        vocabulary.fit_on_words(words)
        indices = vocabulary.words_to_idx(words)
        max_seq = 10
        text_mlp = TextCNN(num_classes=2,
                           embedding_dim=128, embedding_trainable=True, embedding_vocab_size=vocabulary.vocab_size,
                           filter_sizes=[2,3,4,5], num_filters=10, dropout=0.1,
                           num_hidden_units=[10], max_seq_len=max_seq, multi_label=False)
        text_mlp.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
        inputs = pad_sequences(indices, maxlen=max_seq, padding='post', value=0.)
        text_mlp.fit(inputs, labels, epochs=100, batch_size=10)
        y_prob = text_mlp.predict(inputs)
        y_pred = (y_prob >= 0.5).astype(int)
        assert (y_pred == labels).all()

    def test_text_cnn_with_trained_embedding_matrix(self):
        jieba_tokenizer = JiebaTokenizer()
        text = self.text
        labels = np.array([[0, 1], [1, 0]])
        words = jieba_tokenizer.text_to_words(text)
        w2v_embedding = Word2vecEmbedder(dim=128)
        w2v_embedding.fit_on_words(words, min_count=1)
        vocabulary = Vocabulary(w2v_embedding.get_vocabs())
        indices = vocabulary.words_to_idx(words)
        em_matrix = w2v_embedding.build_embedding_matrix()
        max_seq = 10
        text_mlp = TextCNN(num_classes=2,
                           embedding_dim=128, embedding_trainable=True, embedding_vocab_size=vocabulary.vocab_size,
                           embedding_matrix=em_matrix, dropout=0.1,
                           filter_sizes=[2, 3, 4, 5], num_filters=10,
                           num_hidden_units=[], max_seq_len=max_seq, multi_label=False)
        text_mlp.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
        inputs = pad_sequences(indices, maxlen=max_seq, padding='post', value=0.)
        text_mlp.fit(inputs, labels, epochs=100, batch_size=10)
        y_prob = text_mlp.predict(inputs)
        y_pred = (y_prob >= 0.5).astype(int)
        assert (y_pred == labels).all()

    def test_text_rcnn(self):
        jieba_tokenizer = JiebaTokenizer()
        text = self.text
        labels = np.array([[0, 1], [1, 0]])
        words = jieba_tokenizer.text_to_words(text)
        vocabulary = Vocabulary()
        vocabulary.fit_on_words(words)
        indices = vocabulary.words_to_idx(words)
        max_seq = 10
        text_mlp = TextRCNN(num_classes=2,
                            embedding_dim=128, embedding_trainable=True, embedding_vocab_size=vocabulary.vocab_size,
                            rnn_hidden_units=100, conv_hidden_units=100,
                            dropout=0.1,
                            num_hidden_units=[50], max_seq_len=max_seq, multi_label=False)
        text_mlp.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
        inputs = pad_sequences(indices, maxlen=max_seq, padding='post', value=0.)
        text_mlp.fit(inputs, labels, epochs=100, batch_size=10)
        y_prob = text_mlp.predict(inputs)
        y_pred = (y_prob >= 0.5).astype(int)
        assert (y_pred == labels).all()

    def test_text_rcnn_with_trained_embedding_matrix(self):
        jieba_tokenizer = JiebaTokenizer()
        text = self.text
        labels = np.array([[0, 1], [1, 0]])
        words = jieba_tokenizer.text_to_words(text)
        w2v_embedding = Word2vecEmbedder(dim=128)
        w2v_embedding.fit_on_words(words, min_count=1)
        vocabulary = Vocabulary(w2v_embedding.get_vocabs())
        indices = vocabulary.words_to_idx(words)
        em_matrix = w2v_embedding.build_embedding_matrix()
        max_seq = 10
        text_mlp = TextRCNN(num_classes=2,
                           embedding_dim=128, embedding_trainable=True, embedding_vocab_size=vocabulary.vocab_size,
                           embedding_matrix=em_matrix, dropout=0.1,
                            rnn_hidden_units=100, conv_hidden_units=100,
                           num_hidden_units=[], max_seq_len=max_seq, multi_label=False)
        text_mlp.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['acc'])
        inputs = pad_sequences(indices, maxlen=max_seq, padding='post', value=0.)
        text_mlp.fit(inputs, labels, epochs=100, batch_size=10)
        y_prob = text_mlp.predict(inputs)
        y_pred = (y_prob >= 0.5).astype(int)
        assert (y_pred == labels).all()
