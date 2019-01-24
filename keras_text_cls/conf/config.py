import logging
import configparser
from keras_text_cls.conf import CONFIG_FILE


class Config(object):
    def __init__(self, file_path=CONFIG_FILE):
        self.__config = configparser.ConfigParser()
        self.__config.read(file_path)

    def get_config(self):
        return self.__config

    @staticmethod
    def __get_log_level(levels):
        return {
            'logging.INFO': logging.INFO,
            'logging.DEBUG': logging.DEBUG,
            'logging.WARNING': logging.WARNING,
            'logging.ERROR': logging.ERROR,
        }[levels]

    def get_log_level(self):
        return self.__get_log_level(self.__config["log"]["log.level"])
