import keras
import logging
import numpy as np


def init_embedding_layer(embedding_matrix, embedding_dim, embedding_vocab_size, embedding_trainable, max_seq_len,
                         mask_zero=False):
    if embedding_matrix is not None:
        embedding_matrix = np.array(embedding_matrix)
        assert (len(embedding_matrix.shape) == 2)
        assert (embedding_matrix.shape[1] == embedding_dim)
        if embedding_vocab_size is not None:
            # validate the dim of vocabs and embedding matrix
            assert embedding_vocab_size == len(embedding_matrix)
        logging.info("using provided embedding matrix with shape " + str(embedding_matrix.shape))
        layer_embedding = keras.layers.Embedding(len(embedding_matrix),
                                                  embedding_dim,
                                                  weights=[embedding_matrix],
                                                  input_length=max_seq_len,
                                                  trainable=embedding_trainable,
                                                  mask_zero=mask_zero)
    else:
        if embedding_vocab_size is None:
            raise ValueError("embedding_vocab_size must be set a positive integer"
                             " to indicate the input size of the embedding layer"
                             " when embedding_matrix=None")
        if not embedding_trainable:
            raise ValueError("embedding_trainable cannot be false when embedding_matrix=None")
        logging.info("no pre-trained embedding matrix provided, init with shape "
                     + str((embedding_vocab_size, embedding_dim)))
        layer_embedding = keras.layers.Embedding(embedding_vocab_size,
                                                  embedding_dim,
                                                  input_length=max_seq_len,
                                                  trainable=True,
                                                  mask_zero=mask_zero)
    return layer_embedding
