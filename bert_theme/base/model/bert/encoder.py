import math

import tensorflow as tf

from bert_theme.base.model.bert.shape_helper import reshape_to_matrix, create_initializer, dropout, layer_norm
from .activation_helper import get_activation


def encoder_layer(encoder_input,
                  batch_size=4,
                  seq_length=512,
                  num_attention_heads=12,
                  attention_mask=None,
                  hidden_size=768,
                  hidden_dropout_prob=0.1,
                  intermediate_size=3072,
                  intermediate_act=get_activation("gelu"),
                  attention_probs_dropout_prob=0.1,
                  initializer_range=0.02):
    """
    :param attention_mask:
    :param seq_length:
    :param batch_size:
    :param encoder_input: [batch_size * seq_length, hidden_size]
    :param num_attention_heads:
    :param hidden_size:
    :param hidden_dropout_prob:
    :param intermediate_size:
    :param intermediate_act:
    :param attention_probs_dropout_prob:
    :param initializer_range:
    :return: [batch_size * seq_length, embedding_size]
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))
    attention_head_size = int(hidden_size / num_attention_heads)
    with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
            attention_output = attention_layer(
                encoder_input,
                batch_size=batch_size,
                seq_length=seq_length,
                attention_mask=attention_mask,
                num_attention_heads=num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                initializer_range=initializer_range
            )
        with tf.variable_scope("output"):
            attention_output = tf.layers.dense(
                attention_output,
                hidden_size,
                kernel_initializer=create_initializer(initializer_range)
            )
            attention_output = dropout(attention_output, hidden_dropout_prob)
            attention_output = layer_norm(attention_output + encoder_input)
    with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act,
            kernel_initializer=create_initializer(initializer_range)
        )
    with tf.variable_scope("output"):
        encoder_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range)
        )
        encoder_output = dropout(encoder_output, hidden_dropout_prob)
        encoder_output = layer_norm(encoder_output + attention_output)
    return encoder_output


def attention_layer(input_tensor,
                    batch_size=None,
                    seq_length=None,
                    attention_mask=None,
                    num_attention_heads=12,
                    size_per_head=512,
                    attention_probs_dropout_prob=0.1,
                    initializer_range=0.02):
    """
    :param attention_mask:
    :param batch_size:
    :param seq_length:
    :param input_tensor: [batch_size * seq_length, hidden_size]
    :param num_attention_heads:
    :param size_per_head:
    :param attention_probs_dropout_prob:
    :param initializer_range:
    :return: [batch_size * seq_length, hidden_size]
    """

    def transpose_for_scores(
            trans_tensor,
            dimension_0,
            dimension_1,
            dimension_2,
            dimension_3):
        output_tensor = tf.reshape(
            trans_tensor, [dimension_0, dimension_2, dimension_1, dimension_3])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    assert batch_size is not None
    assert seq_length is not None

    # 保证安全，再进行一次变形
    tensor_2d = reshape_to_matrix(input_tensor)  # [batch_size * seq_length, hidden_size]

    query_layer = tf.layers.dense(
        tensor_2d,
        num_attention_heads * size_per_head,
        name="query",
        kernel_initializer=create_initializer(initializer_range)
    )

    key_layer = tf.layers.dense(
        tensor_2d,
        num_attention_heads * size_per_head,
        name="key",
        kernel_initializer=create_initializer(initializer_range)
    )

    value_layer = tf.layers.dense(
        tensor_2d,
        num_attention_heads * size_per_head,
        name="value",
        kernel_initializer=create_initializer(initializer_range)
    )

    query_layer = transpose_for_scores(
        query_layer, batch_size,
        num_attention_heads, seq_length,
        size_per_head)

    key_layer = transpose_for_scores(
        key_layer, batch_size,
        num_attention_heads, seq_length,
        size_per_head)

    value_layer = tf.reshape(
        value_layer,
        [batch_size, seq_length, num_attention_heads, size_per_head]
    )
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
        attention_scores += adder

    attention_probs = tf.nn.softmax(attention_scores)
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    context_layer = tf.matmul(attention_probs, value_layer)
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
    context_layer = tf.reshape(
        context_layer,
        [batch_size * seq_length, num_attention_heads * size_per_head]
    )
    return context_layer
