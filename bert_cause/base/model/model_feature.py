# -*- coding: UTF-8 -*-
import codecs
import collections
import os
import pickle

import tensorflow as tf

from bert_cause.base.bert import tokenization
from bert_cause.base.helper.log_helper import get_logger

logger = get_logger()


class ModelFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids


def example2feature(ex_index, example, label_list, max_seq_length, tokenizer, output_dir):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param label_list:
    :param ex_index: index
    :param example: 一个样本
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :return:
    """
    label_map = {}

    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i

    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    # text_list = example.text
    # label_list = example.label
    # if len(text_list) != len(label_list):
    #     raise ValueError('Your text-split-list dose not match label-split-list, '
    #                      'length of text is: %d and label is: %d' % (len(text_list), len(label_list)))

    tokens = []
    labels = []
    for i, word in enumerate(example.text):
        # 分字
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label = example.label[i]
        for m in range(len(token)):
            labels.append(label)
    # 序列截断
    if len(tokens) > max_seq_length:
        tokens = tokens[0:max_seq_length]
        labels = labels[0:max_seq_length]

    n_tokens = []
    label_ids = []

    for i, token in enumerate(tokens):
        n_tokens.append(token)
        label_ids.append(label_map[labels[i]])

    input_ids = tokenizer.convert_tokens_to_ids(n_tokens)  # 将序列中的字(n_tokens)转化为ID形式
    assert len(input_ids) != 0
    input_mask = [1] * len(input_ids)
    # padding, 使用0填充
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        label_ids.append(0)
        n_tokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % example.guid)
        logger.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logger.info("labels: %s" % " ".join([str(x) for x in labels]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = ModelFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        label_ids=label_ids)
    return feature


def examples2features(examples, max_seq_length, tokenizer):
    """ 从 InputExamples 转化为 ModelFeatures """

    for example in examples:
        tokens = []
        for word in example.text:
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        tokens = tokens[0: max_seq_length]

        input_mask = [1] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        pad_len = max_seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        yield ModelFeatures(
            input_ids=input_ids,
            input_mask=input_mask)


def write_features(examples, label_list, max_seq_length, tokenizer, output_file, output_dir):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param label_list:
    :param output_dir:
    :param examples:  训练样本
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (idx, example) in enumerate(examples):
        if idx % 5000 == 0:
            logger.info("Writing example %d of %d" % (idx, len(examples)))
        # 对于每一个训练样本
        feature = example2feature(idx, example, label_list, max_seq_length, tokenizer, output_dir)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
