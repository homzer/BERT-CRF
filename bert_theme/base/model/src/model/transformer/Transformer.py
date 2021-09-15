import tensorflow as tf

from src.model.rnn.GRU import GRULayer
from src.model.transformer.Decoder import Decoder
from src.model.transformer.Encoder import Encoder
from src.model.transformer.Embedding import Embedding
from src.model.transformer.ProjectLayer import ProjectLayer
from src.utils.TensorUtil import create_attention_mask, create_tensor_mask


def Transformer(input_ids, label_ids, num_encoders=8, num_decoders=8):
    """ Construct Transformer model """

    """ Embeddings """
    encoder_input = Embedding(input_ids)

    """ Encoders """
    with tf.variable_scope("encoder"):
        input_mask = create_tensor_mask(input_ids)
        attention_mask = create_attention_mask(input_ids, input_mask)
        for i in range(num_encoders):
            encoder_output = Encoder(encoder_input, attention_mask, scope="layer_%d" % i)
            encoder_input = encoder_output

    # """ Decoders """
    # with tf.variable_scope("decoder"):
    #     decoder_input
    #     for i in range(num_decoders):
    #         decoder_output = Decoder(
    #             decoder_input,
    #             encoder_output,
    #             forward_attention_mask,
    #             enc_dec_attention_mask,
    #             scope="layer_%d" % i)
    #         decoder_input = decoder_output

    """ GRU Layer """
    with tf.variable_scope("decoder"):
        decoder_output = GRULayer(encoder_output)

    """ Project Layer """
    pred_ids, loss = ProjectLayer(input_tensor=decoder_output, label_ids=label_ids)

    return pred_ids, loss
