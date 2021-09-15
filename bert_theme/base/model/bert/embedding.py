import tensorflow as tf

from bert_theme.base.model.bert.shape_helper import get_shape_list, create_initializer, layer_norm, dropout


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=768,
                     initializer_range=0.02):
    """
    根据输入的 ids 查找 embedding table 从而转化成相应 embedding 表示
    :param input_ids: [batch_size, seq_length]
    :param vocab_size:
    :param embedding_size:
    :param initializer_range:
    :return: [batch_size, seq_length, embedding_size]
    """
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=-1)

    embedding_table = tf.get_variable(
        name="word_embeddings",
        shape=[vocab_size, embedding_size],
        initializer=create_initializer(initializer_range))
    output = tf.nn.embedding_lookup(embedding_table, input_ids)
    input_shape = get_shape_list(input_ids)
    output = tf.reshape(
        output, shape=input_shape[0:-1] + [input_shape[-1] * embedding_size])
    return output


def embedding_position(input_tensor,
                       max_position_embeddings=512,
                       initializer_range=0.02,
                       dropout_prob=0.1):
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    seq_length = input_shape[1]
    embedding_size = input_shape[2]

    output = input_tensor

    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
        full_position_embeddings = tf.get_variable(
            name="position_embeddings",
            shape=[max_position_embeddings, embedding_size],
            initializer=create_initializer(initializer_range))
        position_embeddings = tf.slice(
            full_position_embeddings, [0, 0], [seq_length, -1])
        num_dims = len(output.shape.as_list())
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([seq_length, embedding_size])
        position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
        output += position_embeddings

    def layer_norm_and_dropout(_input_tensor, _dropout_prob, name=None):
        """Runs layer normalization followed by dropout."""
        output_tensor = layer_norm(_input_tensor, name)
        output_tensor = dropout(output_tensor, _dropout_prob)
        return output_tensor

    output = layer_norm_and_dropout(output, dropout_prob)
    return output
