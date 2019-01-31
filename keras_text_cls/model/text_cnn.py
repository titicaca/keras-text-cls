import logging
import keras
from keras_text_cls.model import BaseModel
from keras_text_cls.model.utils import init_embedding_layer
from keras.layers import Dense, Input, Flatten, Dropout, Conv1D, MaxPooling1D, concatenate


class TextCNN(BaseModel):
    """
    Text CNN Model
    """
    def __init__(self, num_classes,
                 embedding_dim=128, embedding_matrix=None, embedding_trainable=False, embedding_vocab_size=None,
                 num_filters=50, filter_sizes=[2,3,4,5],
                 pooling_strategy="REDUCE_MAX", max_seq_len=300,
                 num_hidden_units=[100], hidden_activation="relu",
                 dropout=0.5, multi_label=True):
        super(TextCNN, self).__init__(name='TextCNN')
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.embedding_trainable = embedding_trainable
        self.pooling_strategy = pooling_strategy
        self.max_seq_len = max_seq_len
        self.num_hidden_units = num_hidden_units
        self.hidden_activation = hidden_activation
        self.dropout = dropout
        self.multi_label = multi_label
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        # embedding layer
        self.layer_embedding = init_embedding_layer(embedding_matrix, embedding_dim, embedding_vocab_size,
                                                    embedding_trainable, max_seq_len)

        # tuple list containing (l_conv, l_pool, l_flatten), the output will be concatenated together
        self.layer_convs = []
        for fsz in filter_sizes:
            l_conv = Conv1D(filters=num_filters, kernel_size=fsz, activation='relu')
            l_pool = MaxPooling1D(max_seq_len - fsz + 1)
            l_flatten = Flatten()
            self.layer_convs.append((l_conv, l_pool, l_flatten))

        # dropout between the conv layer and the dense layer
        if dropout > 0:
            self.layer_dropout = keras.layers.Dropout(dropout)

        # hidden layer
        self.layer_hiddens = []
        prev_input_dim = embedding_dim
        for n in num_hidden_units:
            self.layer_hiddens.append(
                keras.layers.Dense(n, input_dim=prev_input_dim, activation=hidden_activation)
            )
            if dropout > 0:
                self.layer_hiddens.append(keras.layers.Dropout(dropout))
            prev_input_dim = n

        if multi_label:
            output_activation = "sigmoid"
        else:
            output_activation = "softmax"
        # output layer
        self.layer_output = keras.layers.Dense(num_classes, activation=output_activation)

    def call(self, inputs):
        """
        :param inputs: 2-dim list, the element is the word index
        :return: predicted class probabilities
        """
        x = self.layer_embedding(inputs)
        convs = []
        for (l_conv, l_pool, l_flatten) in self.layer_convs:
            conv_f = l_conv(x)
            conv_f = l_pool(conv_f)
            conv_f = l_flatten(conv_f)
            convs.append(conv_f)
        x = concatenate(convs, axis=1)
        if self.dropout > 0:
            x = self.layer_dropout(x)
        for hidden in self.layer_hiddens:
            x = hidden(x)
        return self.layer_output(x)

