import tensorflow as tf

from src.utils import ConfigUtil
from src.utils.TensorUtil import reshape2Matrix, get_activation, create_initializer


def ProjectLayer(input_tensor, label_ids, scope=None):
    """
    Compute Cross Entropy Loss
    :param scope:
    :param input_tensor: [batch_size, seq_length, hidden_size] or
    [batch_size * seq_length, hidden_size]
    :param label_ids: [batch_size, seq_length]
    :return: prediction with shape of [batch_size, seq_length] and
    a scalar of loss
    """
    with tf.variable_scope(scope, default_name="softmax"):
        input_tensor = reshape2Matrix(input_tensor)
        with tf.variable_scope("logits"):
            logits = tf.layers.dense(
                inputs=input_tensor,
                units=64,
                activation=get_activation("gelu"),
                kernel_initializer=create_initializer())
            logits = tf.layers.dense(
                inputs=logits,
                units=ConfigUtil.vocab_size,
                activation=get_activation("gelu"),
                kernel_initializer=create_initializer())
            logits = tf.reshape(logits, shape=[-1, ConfigUtil.seq_length, ConfigUtil.vocab_size])
            pred_ids = tf.argmax(logits, -1)
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=label_ids))
    return pred_ids, loss
