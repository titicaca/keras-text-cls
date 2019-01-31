import logging
import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer


class MaskedGlobalAvgPool1D(Layer):
    """
    Masked Global Average Pooling for 1-Dimension Data
    Masked position won't be calculated for avg pooling

    Attributes
    ----------
    axis: int
        the axis for average pooling calculation in the tensor, default 1
    """
    def __init__(self, axis=1, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MaskedGlobalAvgPool1D, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.repeat(mask, x.shape[-1])
            mask = tf.transpose(mask, [0, 2, 1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
        else:
            return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)
