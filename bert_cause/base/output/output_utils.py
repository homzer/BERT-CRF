# -*- coding: UTF-8 -*-
from bert_cause.base.helper.label_helper import ConstantLabel as ConLabel


def id_to_label(lbl_ids, id2label):
    """
    将 id 转化为标签
    :param lbl_ids: [max_seq_length, ] 一维向量
    :param id2label: 标签字典
    :return: 一维标签列表
    """
    labels = []
    for i, id in enumerate(lbl_ids):
        try:
            label = id2label[id]
        except KeyError:
            labels.append(ConLabel.X)
            continue
        labels.append(label)
    return labels


def get_predict_labels(pred_ids, id2label):
    """
     获取每句话的预测标签
    :param pred_ids: 标签二维列表 [num_lines, max_seq_length]
    :param id2label: 标签字典
    :return: 转化为字符标签的二维列表
    """
    predict_labels = []
    for pred in pred_ids:
        labels = id_to_label(pred, id2label)
        predict_labels.append(labels)
    return predict_labels


def get_context_tokens(predict_examples):
    """
    获取所有句子经过分词的列表
    :param predict_examples: InputExample 类的实例的列表
    :return: 二维列表
    """
    context_tokens = []
    for predict_line in predict_examples:
        line_tokens = predict_line.text
        context_tokens.append(line_tokens)
    return context_tokens


def get_context_labels(predict_examples):
    """
    获取所有句子标签经过的列表
    :param predict_examples: InputExample 类的实例的列表
    :return: 二维列表
    """
    context_labels = []
    for predict_line in predict_examples:
        line_label = predict_line.label
        context_labels.append(line_label)
    return context_labels

