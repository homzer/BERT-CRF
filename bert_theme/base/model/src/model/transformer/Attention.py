import math

import tensorflow as tf

from bert_theme.base.model.src.utils.TensorUtil import reshape2Matrix, create_initializer, dropout, layer_norm


def Attention(
        from_tensor,
        to_tensor=None,
        attention_mask=None,
        num_attention_heads=12):
    """
    Construct Attention Layer, if to_tensor is None, compute Self-Attention.
    :param num_attention_heads: number of heads.
    :param from_tensor: [batch_size, from_seq_length, hidden_size]
    :param to_tensor: [batch_size, to_seq_length, hidden_size]
    :param attention_mask: [batch_size, from_seq_length, to_seq_length]
    :return: attention_output with shape of [batch_size, from_seq_length * hidden_size]
    """
    to_tensor = from_tensor if to_tensor is None else to_tensor
    hidden_size = from_tensor.shape[2]
    from_seq_length = from_tensor.shape[1]
    to_seq_length = to_tensor.shape[1]
    assert int(hidden_size) % num_attention_heads == 0
    size_per_head = int(int(hidden_size) / num_attention_heads)

    def create_layer(input_tensor, name):
        return tf.layers.dense(
            input_tensor, num_attention_heads * size_per_head,
            name=name, activation=None,
            kernel_initializer=create_initializer())

    def transpose_for_scores(input_tensor, seq_length):
        """
        :param input_tensor: shape of [batch_size * seq_length, num_attention_heads * size_per_head],
        reshape tensor to [batch, seq_length, num_attention_heads, size_per_head] and
        transpose tensor [0, 1, 2, 3] to [0, 2, 1, 3]
        :param seq_length: the length of sequence,
        which specifically, can be `from_seq_length` or `to_seq_length`
        :return [batch_size, num_attention_heads, seq_length, size_per_head]
        """
        output_tensor = tf.reshape(
            input_tensor,
            [-1, seq_length, num_attention_heads, size_per_head])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def scaled_dot_product(q, k, v):
        """
        Apply scaled-dot product.
        :return: [batch, from_seq_length, num_attention_heads, size_per_head]
        """
        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(size_per_head))

        if attention_mask is not None:
            # [batch_size, 1, from_seq_length, to_seq_length]
            expanded_attention_mask = tf.expand_dims(attention_mask, axis=[1])
            adder = (1.0 - tf.cast(expanded_attention_mask, tf.float32)) * -10000.0
            attention_scores += adder

        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = dropout(attention_probs)
        scaled_output = tf.matmul(attention_probs, v)
        scaled_output = tf.transpose(scaled_output, [0, 2, 1, 3])
        return scaled_output, attention_probs

    with tf.variable_scope("attention"):
        from_tensor = reshape2Matrix(from_tensor)
        to_tensor = reshape2Matrix(to_tensor)
        query = create_layer(from_tensor, 'query')
        key = create_layer(to_tensor, 'key')
        value = create_layer(to_tensor, 'value')
        query = transpose_for_scores(query, from_seq_length)
        key = transpose_for_scores(key, to_seq_length)
        value = transpose_for_scores(value, to_seq_length)

        attention_output, probs = scaled_dot_product(query, key, value)
        attention_output = tf.reshape(
            attention_output, [-1, num_attention_heads * size_per_head])

        with tf.variable_scope("output"):
            attention_output = tf.layers.dense(
                attention_output, hidden_size,
                kernel_initializer=create_initializer())
            attention_output = dropout(attention_output)
            attention_output = layer_norm(attention_output + from_tensor)
    return attention_output, probs
