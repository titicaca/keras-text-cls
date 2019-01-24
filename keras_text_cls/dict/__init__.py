import os


DICT_DIR = os.path.dirname(os.path.abspath(__file__))
JIEBA_DICT_PATH = os.path.join(DICT_DIR, "jieba_dict")
STOP_WORDS_PATH = os.path.join(DICT_DIR, "stop_words")


def load_stop_words(self, filename=STOP_WORDS_PATH):
    with open(filename, 'rb') as f:
        _stop_words = set([line.strip().decode('utf-8') for line in f.readlines()])
    return _stop_words
