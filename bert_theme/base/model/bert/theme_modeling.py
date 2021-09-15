import copy
import tensorflow as tf

from .shape_helper import reshape_to_matrix, get_shape_list, reshape_from_matrix
from .encoder import encoder_layer
from .activation_helper import get_activation
from .embedding import embedding_lookup, embedding_position


def create_attention_mask(input_tensor, input_mask):
    """
    :param input_tensor: [batch_size, seq_length, hidden_size]
    :param input_mask: [batch_size, seq_length]
    :return: [batch_size, seq_length, seq_length]
    """
    tensor_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = tensor_shape[0]
    seq_length = tensor_shape[1]

    input_mask = tf.cast(
        tf.reshape(input_mask, [batch_size, 1, seq_length]), tf.float32
    )

    broadcast_ones = tf.ones(
        shape=[batch_size, seq_length, 1], dtype=tf.float32
    )

    mask = broadcast_ones * input_mask
    return mask


class BertModel(object):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 num_encoders=4,
                 scope=None):
        """

        :param config:
        :param is_training:
        :param input_ids: [batch_size, seq_length]
        :param input_mask:
        :param num_encoders:
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                self.embedding_output = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range)

                self.embedding_output = embedding_position(
                    input_tensor=self.embedding_output,
                    max_position_embeddings=config.max_position_embeddings,
                    initializer_range=config.initializer_range,
                    dropout_prob=config.hidden_dropout_prob)

            with tf.variable_scope("transformer"):
                # [batch_size, seq_length, seq_length]
                attention_mask = create_attention_mask(self.embedding_output, input_mask)
                self.all_encoder_layers = []
                prev_output = reshape_to_matrix(self.embedding_output)  # [batch_size * seq_length, embedding_size]
                for idx in range(num_encoders):
                    with tf.variable_scope("encoder_%d" % idx):
                        encoder_output = encoder_layer(
                            encoder_input=prev_output,
                            batch_size=batch_size,
                            seq_length=seq_length,
                            attention_mask=attention_mask,
                            num_attention_heads=config.num_attention_heads,
                            hidden_size=config.hidden_size,
                            hidden_dropout_prob=config.hidden_dropout_prob,
                            intermediate_size=config.intermediate_size,
                            intermediate_act=get_activation(config.hidden_act),
                            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                            initializer_range=config.initializer_range
                        )
                        self.all_encoder_layers.append(encoder_output)
                        prev_output = encoder_output
                self.transformer_output = reshape_from_matrix(
                    self.all_encoder_layers[-1], [batch_size, seq_length, -1])

    def get_transformer_output(self):
        """ :return [batch_size, seq_length, embedding_size] """
        return self.transformer_output
