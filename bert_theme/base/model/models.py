# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

from bert_theme.base.model.src.model.transformer.Embedding import Embedding
from bert_theme.base.model.src.model.transformer.Encoder import Encoder
from bert_theme.base.model.src.utils.TensorUtil import create_tensor_mask, create_attention_mask


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
    embeds_mix = tf.reshape(embeds_mix, shape=[-1, 2 * 768])
    return embeds_mix


def create_model(bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 themes,
                 num_themes,
                 dropout_rate=1.0):
    """
    创建X模型
    :param num_themes:
    :param themes: [batch_size, num_themes]
    :param dropout_rate:
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :return:
    """
    embeddings = Embedding(input_ids)
    with tf.variable_scope("encoder"):
        input_mask = create_tensor_mask(input_ids)
        attention_mask = create_attention_mask(input_ids, input_mask)
        encoder_output, _ = Encoder(embeddings, attention_mask, scope='layer_0')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_1')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_2')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_3')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_4')
        encoder_output, _ = Encoder(encoder_output, attention_mask, scope='layer_5')
    with tf.variable_scope("theme"):
        concat_embeds = max_and_mean_concat(encoder_output, input_mask)
        with tf.variable_scope("logits"):
            w = tf.get_variable(
                'w', shape=[768 * 2, 4],
                dtype=tf.float32, initializer=initializers.xavier_initializer())
            b = tf.get_variable(
                'b', shape=[4], dtype=tf.float32,
                initializer=tf.zeros_initializer())
            logits = tf.tanh(tf.nn.xw_plus_b(concat_embeds, w, b))  # [batch_size, num_themes]
            # 获取最大下标，得到预测值
            predicts = tf.argmax(logits, -1)
        with tf.variable_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=themes))

    prediction_dict = {"embeddings": encoder_output, "pred_theme": predicts}
    return loss, prediction_dict, logits


def get_max_length(embedding):
    return embedding.shape[1].value


def get_used_length(input_ids):
    # 算序列真实长度 sign(-2.3)=-1.0, sign(45)=1, sign(0)=0
    used = tf.sign(tf.abs(input_ids))
    # reduce_sum([[2,3,1],[3,3,1]], indices=0)=[5,6,2] 或者当 indices=1 为 [6,7]
    lengths = tf.reduce_sum(used, reduction_indices=1)
    return lengths
