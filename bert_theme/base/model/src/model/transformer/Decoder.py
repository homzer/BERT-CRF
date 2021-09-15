import tensorflow as tf

from bert_theme.base.model.src.model.transformer.Attention import Attention
from bert_theme.base.model.src.utils import ConfigUtil
from bert_theme.base.model.src.utils.TensorUtil import create_initializer, layer_norm, dropout, get_activation


def Decoder(
        input_tensor,
        encoder_output,
        forward_attention_mask,
        enc_dec_attention_mask,
        scope=None):
    """ Decoder Layer """
    hidden_size = input_tensor.shape[-1]
    with tf.variable_scope(scope, default_name="decoder"):
        """ Self-Attention with forward Attention mask"""
        attention_output = Attention(input_tensor, attention_mask=forward_attention_mask)

        with tf.variable_scope("addNorm1"):
            attention_output = tf.layers.dense(
                attention_output, hidden_size,
                kernel_initializer=create_initializer())
            attention_output = dropout(attention_output, ConfigUtil.dropout_prob)
            attention_output = layer_norm(attention_output + input_tensor)

        """ Enc-Dec Attention Layer """
        enc_dec_attention_output = Attention(
            attention_output, encoder_output, enc_dec_attention_mask)

        with tf.variable_scope("addNorm2"):
            enc_dec_attention_output = tf.layers.dense(
                enc_dec_attention_output, hidden_size,
                kernel_initializer=create_initializer())
            enc_dec_attention_output = dropout(enc_dec_attention_output, ConfigUtil.dropout_prob)
            enc_dec_attention_output = layer_norm(enc_dec_attention_output + attention_output)

        with tf.variable_scope("intermediate"):
            intermediate_output = tf.layers.dense(
                enc_dec_attention_output, ConfigUtil.intermediate_size,
                activation=get_activation(ConfigUtil.activation),
                kernel_initializer=create_initializer())

        with tf.variable_scope("addNorm3"):
            decoder_output = tf.layers.dense(
                intermediate_output, hidden_size,
                kernel_initializer=create_initializer())
            decoder_output = dropout(decoder_output)
            decoder_output = layer_norm(decoder_output + enc_dec_attention_output)
    return decoder_output
