import tensorflow as tf

from bert_theme.base.model.src.model.transformer.Attention import Attention
from bert_theme.base.model.src.utils.TensorUtil import create_initializer, dropout, layer_norm, get_activation, reshape2Matrix

__all__ = ['Encoder']


def Encoder(
        input_tensor,
        attention_mask=None,
        scope=None):
    """
     Construct encoder layer
    :param input_tensor: `Tensor` with shape of [batch_size, seq_length, hidden_size]
    :param attention_mask: `Tensor` [batch, seq_length, seq_length]
    :param scope:
    :return: `Tensor` with same shape to `input_tensor`
    """
    seq_length = input_tensor.shape[1]
    hidden_size = input_tensor.shape[2]
    with tf.variable_scope(scope, default_name="encoder"):
        attention_output, probs = Attention(input_tensor, attention_mask=attention_mask)

        with tf.variable_scope("intermediate"):
            intermediate_output = tf.layers.dense(
                attention_output, units=3072,
                activation=get_activation('gelu'),
                kernel_initializer=create_initializer())

        with tf.variable_scope("output"):
            encoder_output = tf.layers.dense(
                intermediate_output, hidden_size,
                kernel_initializer=create_initializer())
            encoder_output = dropout(encoder_output)
            encoder_output = layer_norm(encoder_output + attention_output)
    return tf.reshape(encoder_output, shape=[-1, seq_length, hidden_size]), probs
