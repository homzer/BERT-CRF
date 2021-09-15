import os

import tensorflow as tf

INFO = 1
WARN = 2
ERROR = 3


def set_verbosity(level=ERROR):
    if level == ERROR:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    elif level == WARN:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    else:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
