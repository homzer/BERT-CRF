# -*- coding: utf-8 -*-

from bert_cause.base.model.classify.lstm_crf_layer import BlstmCrf
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow as tf


def create_model(bert_config, is_training, input_ids, input_mask,
                 labels, num_labels, use_one_hot_embeddings,
                 dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):
    """
    创建X模型
    :param num_layers:
    :param cell:
    :param dropout_rate:
    :param lstm_size:
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param labels: 标签的idx 表示 [batch_size, seq_length]
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    from bert_cause.base.model.bert import modeling

    # 创建bert模型
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    model_output = model.get_sequence_output()  # [batch_size, seq_length, embedding_size]
    blstm_crf = BlstmCrf(embedded_chars=model_output, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                         dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                         seq_length=get_max_length(model_output), labels=labels, lengths=get_used_length(input_ids),
                         is_training=is_training)
    loss, _, _, pred_ids = blstm_crf.add_blstm_crf_layer(crf_only=True, name='main')

    prediction_dict = {"pred_ids": pred_ids, "embeddings": model_output}
    return loss, prediction_dict


def get_max_length(embedding):
    return embedding.shape[1].value


def get_used_length(input_ids):
    # 算序列真实长度 sign(-2.3)=-1.0, sign(45)=1, sign(0)=0
    used = tf.sign(tf.abs(input_ids))
    # reduce_sum([[2,3,1],[3,3,1]], indices=0)=[5,6,2] 或者当 indices=1 为 [6,7]
    lengths = tf.reduce_sum(used, reduction_indices=1)
    return lengths
