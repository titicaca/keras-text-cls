import logging
import numpy as np
import keras
from keras_text_cls.model.base_model import BaseModel
from keras_text_cls.model.utils import init_embedding_layer
from keras_text_cls.layer import MaskedGlobalAvgPool1D


class TextMLP(BaseModel):
    """
    Multiple Layer Perceptron for Text Classification

    Attributes
    ----------
    num_classes: int
        the number of classes
    embedding_dim: int
        the dimension of embedding vector, default is 128
    embedding_matrix: 2d np.array
        pre-trained embedding matrix is an array of embedding vector,
            where index 0 must be reserved for SYMBOL_PADDING, index 1 must be reserved for SYMBOL_UNKNOWN
        default is None
    embedding_trainable: bool
        Is the embedding layer trainable in the network. It must be set to True, when embedding matrix is None.
        Default is False.
        Set False if embedding matrix is pre-trained and set in the model
    embedding_vocab_size: int
        the vocabulary size for embedding.
        Default is None, which indicates the size is equal to the length of embedding matrix
        embedding_vocab_size must be set to initialize the size of the embedding layer, when embedding_matrix=None
    pooling_strategy: str
        pooling strategy for word sequences, either "REDUCE_MEAN" or "REDUCE_MAX"
    max_seq_len: int
        maximum length of words for each text, longer text will be truncated more than max_seq_len,
        shorter text will be padded
    num_hidden_units: integer array
        an array of positive integers, indicating the number of units for each hidden layer
    hidden_activation: str
        activation function of neutral unit, default is "relu"
    dropout: float (0,1)
        dropout rate, must be equal or greater than 0 and equal or less than 1, default is 0.5
    multi_label: bool
        is the labels are multi-label classification, default is True
    """
    def __init__(self, num_classes,
                 embedding_dim=128, embedding_matrix=None, embedding_trainable=False, embedding_vocab_size=None,
                 pooling_strategy="REDUCE_MEAN", max_seq_len=300,
                 num_hidden_units=[100], hidden_activation="relu",
                 dropout=0.5, multi_label=True):
        super(TextMLP, self).__init__(name='TextMLP')

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

        self.layer_embedding = init_embedding_layer(embedding_matrix, embedding_dim, embedding_vocab_size,
                                                    embedding_trainable, max_seq_len, mask_zero=True)

        if pooling_strategy == "REDUCE_MEAN":
            self.layer_pooling = MaskedGlobalAvgPool1D()
        elif pooling_strategy == "REDUCE_MAX":
            self.layer_pooling = keras.layers.MaxPool1D()
        else:
            raise ValueError("Unknown pooling strategy, only REDUCE_MEAN, REDUCE_MAX are supported")

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
        self.layer_output = keras.layers.Dense(num_classes, activation=output_activation)

    def call(self, inputs):
        """
        :param inputs: 2-dim list, the element is the word index
        :return: predicted class probabilities
        """
        x = self.layer_embedding(inputs)
        x = self.layer_pooling(x)
        for hidden in self.layer_hiddens:
            x = hidden(x)
        return self.layer_output(x)
