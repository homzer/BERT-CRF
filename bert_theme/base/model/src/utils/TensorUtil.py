import tensorflow as tf
from tensorflow.contrib.layers import layer_norm as ln


def reshape2Matrix(input_tensor):
    """ Reshape tensor to a rank 2 tensor """
    width = input_tensor.shape[-1]
    return tf.reshape(input_tensor, [-1, width])


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def dropout(input_tensor, dropout_prob=0.1):
    """ Perform dropout """
    return tf.nn.dropout(input_tensor, 1.0 - dropout_prob)


def layer_norm(input_tensor):
    """Run layer normalization on the last dimension of the tensor."""
    return ln(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1)


def create_attention_mask(from_tensor, to_mask):
    """ Create 3D attention mask from a 2D tensor mask.
    Args:
        from_tensor: 2D Tensor of shape [batch_size, from_seq_length].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    assert from_tensor.shape.ndims == 2
    # `to_mask` = [batch_size, 1, to_seq_length]
    to_mask = tf.expand_dims(to_mask, axis=1)
    to_mask = tf.cast(to_mask, tf.float32)
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones_like(from_tensor, dtype=tf.float32)
    broadcast_ones = tf.expand_dims(broadcast_ones, axis=-1)
    mask = broadcast_ones * to_mask
    return mask


def gelu(input_tensor):
    """ Gaussian Error Linear Unit. """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """ Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`. """
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    elif act == "none":
        return None
    else:
        raise ValueError("Unsupported activation: %s" % act)


def create_tensor_mask(input_tensor, dtype=tf.int32):
    """
    create mask according to input_tensor
    non-0 donates valid, 0 donates invalid
    input_tensor:
    [[2, 3, 1, 0, 0], [5, 0, 0, 0, 0]]
    result:
    [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]]
    :param dtype:
    :param input_tensor: 2D tensor
    :return: input_mask, same shape of input_tensor
    """
    tensor_mask = tf.sign(tf.abs(input_tensor))
    return tf.cast(tensor_mask, dtype=dtype)


def max_and_mean_concat(embeddings, input_mask):
    """
    根据掩码计算embeddings最后一维的平均值和最大值，并将其连接
    :param embeddings: [batch_size, seq_length, embedding_size]
    :param input_mask: [batch_size, seq_length] 1 为有效， 0 为无效
    :return: embeds_mix [batch_size, embedding_size * 2]
    """
    input_mask = tf.cast(input_mask, dtype=tf.float32)
    lengths = tf.reduce_sum(input_mask, axis=-1, keepdims=True)  # [batch_size, 1]
    # 根据掩码对 embeddings 后面不需要部分置零
    embeddings = embeddings * tf.expand_dims(input_mask, axis=-1)
    # 求和取平均
    embeds_mean = tf.reduce_sum(embeddings, axis=1) / lengths  # [batch_size, embedding_size]
    # 求最大值
    embeds_max = tf.reduce_max(embeddings, axis=1)  # [batch_size, embedding_size]
    # 交叉连接
    embeds_mean = tf.expand_dims(embeds_mean, axis=-1)
    embeds_max = tf.expand_dims(embeds_max, axis=-1)
    embeds_mix = tf.concat([embeds_mean, embeds_max], axis=-1)  # [batch_size, embedding_size, 2]
    embeds_mix = tf.reshape(embeds_mix, shape=[embeds_mix.shape[0], -1])
    return embeds_mix
