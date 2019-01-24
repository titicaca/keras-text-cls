import logging
from keras_text_cls.conf import Config

level = Config().get_log_level()

logging.basicConfig(format='%(levelname)s: %(message)s - %(asctime)s - %(pathname)s[line:%(lineno)d] ',
                    level=level)

