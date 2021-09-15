import codecs
import collections
import os
import pickle

import tensorflow as tf

from bert_theme.base.bert import tokenization
from bert_theme.base.helper.log_helper import get_logger

logger = get_logger()


def one_hot_encoding(label_id, n_classes):
    """
    one-hot 编码
    :param label_id: 标签的id表示，并保证 id < n_classes
    :param n_classes: 标签的种类数
    :return: [seq_length * n_classes]
    """
    from keras.utils import np_utils
    encoded = np_utils.to_categorical(label_id, n_classes, dtype=int)
    return encoded


class ModelFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, theme_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.theme_id = theme_id


def example2feature(ex_index, example, theme_list, max_seq_length, tokenizer, output_dir):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param theme_list:
    :param ex_index: index
    :param example: 一个样本
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :return:
    """
    theme_map = {}

    for (i, theme) in enumerate(theme_list):
        theme_map[theme] = i
    theme_id = theme_map[example.theme]
    theme_id = one_hot_encoding(theme_id, len(theme_list))  # 类似于 [0, 1, 0]

    # 保存 theme->index 的map
    if not os.path.exists(os.path.join(output_dir, 'theme2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'theme2id.pkl'), 'wb') as w:
            pickle.dump(theme_map, w)

    text_split_list = example.text.split(' ')

    tokens = []
    for i, word in enumerate(text_split_list):
        # 分字
        token = tokenizer.tokenize(word)
        tokens.extend(token)
    # 序列截断
    if len(tokens) > max_seq_length:
        tokens = tokens[0:max_seq_length]

    n_tokens = []

    for i, token in enumerate(tokens):
        n_tokens.append(token)

    input_ids = tokenizer.convert_tokens_to_ids(n_tokens)  # 将序列中的字(n_tokens)转化为ID形式
    assert len(input_ids) != 0
    input_mask = [1] * len(input_ids)
    # padding, 使用0填充
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        n_tokens.append("**NULL**")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % example.guid)
        logger.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("theme: %s" % example.theme)
        logger.info("theme_id: %s" % " ".join([str(x) for x in theme_id]))

    feature = ModelFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        theme_id=theme_id)
    return feature


def examples2features(examples, max_seq_length, tokenizer):
    """ 从 InputExamples 转化为 ModelFeatures """

    def tokenize_text(text):
        text = text[0: max_seq_length]  # 序列截断
        _tokens = []
        for word in text:
            token = tokenizer.tokenize(word)
            _tokens.extend(token)
        return _tokens

    for example in examples:
        tokens = tokenize_text(example.text)

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


def write_features(
        examples, theme_list, max_seq_length, tokenizer, output_file, output_dir):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param theme_list:
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
        feature = example2feature(idx, example, theme_list, max_seq_length, tokenizer, output_dir)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["theme_id"] = create_int_feature(feature.theme_id)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
