import logging
import keras
from keras_text_cls.model import BaseModel
from keras_text_cls.model.utils import init_embedding_layer
from keras_text_cls.layer import AttentionLayer
from keras.layers import Dense, Dropout, GRU, LSTM, Bidirectional, Input, TimeDistributed
from keras.models import Model


class TextHAN(BaseModel):
    """
    Text RCNN Model
    """
    def __init__(self, num_classes,
                 embedding_dim=128, embedding_matrix=None, embedding_trainable=False, embedding_vocab_size=None,
                 word_rnn_hidden_units=100, sent_rnn_hidden_units=100, rnn_type="GRU",
                 max_sentence_len=100, max_num_sentence=15,
                 dropout=0.5, multi_label=True):
        super(TextHAN, self).__init__(name='TextRCNN')
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.embedding_trainable = embedding_trainable
        self.max_sentence_len = max_sentence_len
        self.max_num_sentence = max_num_sentence
        self.dropout = dropout
        self.multi_label = multi_label
        self.word_rnn_hidden_units = word_rnn_hidden_units
        self.sentence_rnn_hidden_units = sent_rnn_hidden_units
        self.rnn_type = rnn_type

        if rnn_type.upper() == "GRU":
            recurrent = GRU
        elif rnn_type.upper() == "LSTM":
            recurrent = LSTM
        else:
            raise ValueError("param rnn_type can be only either GRU or LSTM")

        layer_sent_input = Input(shape=(max_sentence_len,), dtype='int32')

        # embedding layer
        layer_embedding = init_embedding_layer(embedding_matrix, embedding_dim, embedding_vocab_size,
                                               embedding_trainable, max_sentence_len)(layer_sent_input)

        layer_birnn_word = Bidirectional(recurrent(word_rnn_hidden_units, return_sequences=True))(layer_embedding)
        layer_att_word = AttentionLayer(max_sentence_len)(layer_birnn_word)
        sent_encoder = Model(layer_sent_input, layer_att_word)

        layer_doc_input = Input(shape=(max_num_sentence, max_sentence_len), dtype='int32')
        doc_encoder = TimeDistributed(sent_encoder)(layer_doc_input)
        layer_birnn_sent = Bidirectional(recurrent(sent_rnn_hidden_units, return_sequences=True))(doc_encoder)
        layer_att_sent = AttentionLayer(max_num_sentence)(layer_birnn_sent)
        if multi_label:
            output_activation = "sigmoid"
        else:
            output_activation = "softmax"
        # output layer
        layer_output = keras.layers.Dense(num_classes, activation=output_activation)(layer_att_sent)
        self.model = Model(layer_doc_input, layer_output)

    def call(self, inputs):
        """
        :param inputs: 3-dim np.array, the element is the word index
        :return: predicted class probabilities
        """
        assert len(inputs.shape) == 3
        x = self.model(inputs)
        return x
