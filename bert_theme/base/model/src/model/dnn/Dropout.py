import tensorflow as tf


def Dropout(input_tensor, dropout_prob=0.1):
    """ Perform dropout """
    return tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
