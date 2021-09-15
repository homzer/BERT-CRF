import tensorflow as tf

from bert_theme.base.model.src.utils import ConfigUtil
from bert_theme.base.model.src.utils.TensorUtil import create_initializer, dropout, layer_norm

__all__ = ['Embedding']


def Embedding(input_ids, positional=True):
    """
     Construct embedding layer
    :param input_ids: Tensor of shape [batch_size, seq_length]
    :param positional: whether to add the positional information.
    :return: float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
        embedding_output = embedding_lookup(input_ids, ConfigUtil.vocab_size)
        if positional:
            embedding_output = position_embedding(embedding_output)
    return embedding_output


def embedding_lookup(input_ids, vocab_size=21128, embedding_size=768):
    """
    Looks up words embeddings for id tensor.
    Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word ids.
    Returns:
        float Tensor of shape [batch_size, seq_length, embedding_size].
    """
    assert input_ids.shape.ndims == 2
    embedding_table = tf.get_variable(
        name="word_embeddings",
        shape=[vocab_size, embedding_size],
        initializer=create_initializer())
    output = tf.nn.embedding_lookup(embedding_table, input_ids)
    return output


def position_embedding(input_tensor, max_position_embeddings=512):
    """ Adding position information to input embeddings. """
    seq_length = input_tensor.shape[1]
    width = input_tensor.shape[2]
    output = input_tensor

    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
        full_position_embeddings = tf.get_variable(
            name="position_embeddings",
            shape=[max_position_embeddings, width],
            initializer=create_initializer())
        position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                       [seq_length, -1])
        num_dims = len(output.shape.as_list())
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([seq_length, width])
        position_embeddings = tf.reshape(position_embeddings,
                                         position_broadcast_shape)
        output += position_embeddings
    output = layer_norm(output)
    output = dropout(output)
    return output
