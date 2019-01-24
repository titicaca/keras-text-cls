import keras
from abc import ABCMeta, abstractmethod


class BaseModel(keras.Model):
    __metaclass__ = ABCMeta

    def __init__(self, name="BaseModel"):
        super(BaseModel, self).__init__(name=name)

