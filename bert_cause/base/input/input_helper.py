# -*- coding: UTF-8 -*-
import codecs
import json

from bert_cause.base.bert import tokenization
from bert_cause.base.helper.label_helper import ConstantLabel as ConLabel

DOCUMENT = 'document'
TEXT = 'text'
QAS = 'qas'
OTHER = '其他'

B_CUN = ConLabel.B_CUN
B_CUS = ConLabel.B_CUS
B_CEN = ConLabel.B_CEN
B_EFN = ConLabel.B_EFN
B_EFS = ConLabel.B_EFS
B_EFCUN = ConLabel.B_EFCUN
B_EFCUS = ConLabel.B_EFCUS
I_CUN = ConLabel.I_CUN
I_CUS = ConLabel.I_CUS
I_CEN = ConLabel.I_CEN
I_EFN = ConLabel.I_EFN
I_EFS = ConLabel.I_EFS
I_EFCUN = ConLabel.I_EFCUN
I_EFCUS = ConLabel.I_EFCUS

O = ConLabel.O
B = ConLabel.B
I = ConLabel.I
E = ConLabel.E
S = ConLabel.S
CONNECTOR = ConLabel.CONNECTOR


def read_file(input_file, max_seq_length):
    """
    读取 json 文件，将其中的文本和标签读取出来
    :param input_file: json 文件
    :param max_seq_length: 最大序列长度
    :return: 以 [[标签序列], [文本序列]] 为单位组成的三维列表
    """
    with codecs.open(input_file, 'r', encoding='utf-8') as file:
        content_labels = []
        content_tokens = []
        for line in file:
            try:
                content_dict = json.loads(line)  # 将 json 格式的字符串转化为字典
            except json.decoder.JSONDecodeError:
                continue
            qas = content_dict[QAS]
            if len(qas) == 0:
                continue
            document = content_dict[DOCUMENT][0]
            text = tokenization.convert_to_unicode(document[TEXT])
            text_labels_lists = []
            for qa in qas:
                label = text2label(len(text), qa)
                label = label[0:max_seq_length]  # 序列截断
                text_labels_lists.append(label)
            if len(text_labels_lists) == 0:
                continue
            text = text[0:max_seq_length]  # 序列截断
            labels, tokens = check_overlapping(text_labels_lists, text)
            content_labels.append(labels)
            content_tokens.append(tokens)
        return content_tokens, content_labels


def read_text(content, max_seq_length):
    """
    从文本中读取
    :param content: 文本 字符串列表
    :param max_seq_length: 最大长度
    :return: 文本列表
    """
    if not isinstance(content, type([str])):
        raise TypeError('content must be string list!')
    content_texts = []
    for line in content:
        content_texts.append(line[0: max_seq_length])  # 序列截断
    return content_texts


def wash(text):
    """ 清洗文本 """
    text = str(text).replace(u'\x20', 'O')
    text = str(text).replace(u'\xa0', 'O')
    text = str(text).replace('\u2003', 'O')
    return text


def check_overlapping(labels_lists, text):
    """
    此函数主要针对一句话具有多个原因结果对的情况，将其标在一个句子上，目前可以处理的情况有：
    因果关系对并列、共因不同果、因果叠加的情况。
    其中因果叠加则采用的标签为: B-CUEF, I-CUEF; 仅适用于结果被原因覆盖
    :param labels_lists: 标签的标注序列二维列表
    :param text: 文本
    :return: [标签序列], [文本序列]
    """
    tokens = list(text)
    if len(labels_lists) == 1:
        return labels_lists[0], tokens

    base_label = []
    base_label.extend(labels_lists[0])
    for next_label in labels_lists:
        for i in range(len(next_label)):
            base_tap = base_label[i]
            next_tap = next_label[i]
            if base_tap == O:
                base_label[i] = next_label[i]
            elif (base_tap == I_EFN and next_tap == I_CUN) or \
                    (base_tap == I_CUN and next_tap == I_EFN):
                base_label[i] = I_EFCUN
            elif (base_tap == I_EFS and next_tap == I_CUS) or \
                    (base_tap == I_CUS and next_tap == I_EFS):
                base_label[i] = I_EFCUS
            elif (base_tap == B_EFN and next_tap == B_CUN) or \
                    (base_tap == B_CUN and next_tap == B_EFN):
                base_label[i] = B_EFCUN
            elif (base_tap == B_EFS and next_tap == B_CUS) or \
                    (base_tap == B_CUS and next_tap == B_EFS):
                base_label[i] = B_EFCUS
    return base_label, tokens


def text2label(text_len, qa):
    """
    将文本根据标注转化成 标签序列
    原因中的核心名词：{'B-CUN', 'I-CUN'}
    原因中的谓语或状态：{'B-CUS', 'I-CUS'}
    中心词：{'B-CEN', 'I-CEN'}
    结果中的核心名词：{'B-EFN', 'I-EFN'}
    结果中的谓语或状态：{'B-EFS', 'I-EFS'}
    其他标识：{'O'}
    :param text_len: 文本长度
    :param qa: 标注
    :return: 返回标签序列
    """
    labels = []
    for i in range(text_len):
        labels.append('O')
    for item_dict in qa:
        if 'question' not in item_dict.keys() or 'answers' not in item_dict.keys():
            continue
        question = item_dict['question']
        answers = item_dict['answers']
        if question == '原因中的核心名词':
            start_tab = B_CUN
            end_tab = I_CUN
        elif question == '原因中的谓语或状态':
            start_tab = B_CUS
            end_tab = I_CUS
        elif question == '中心词':
            start_tab = B_CEN
            end_tab = I_CEN
        elif question == '结果中的核心名词':
            start_tab = B_EFN
            end_tab = I_EFN
        elif question == '结果中的谓语或状态':
            start_tab = B_EFS
            end_tab = I_EFS
        else:
            raise Exception('未知的问题名称，无法匹配！ question = %s' % question)
        for answer in answers:
            if len(answer) == 0 or len(answer[TEXT]) == 0:  # 为空则跳过
                continue
            start_idx = answer['start']
            end_idx = start_idx + len(answer[TEXT])
            labels[start_idx] = start_tab
            for idx in range(start_idx + 1, end_idx):
                labels[idx] = end_tab
    return labels


def tag_tail(tag, get_none=True):
    """ 返回一个标签的尾部，必须是 X-Y 的形式，否则返回 None 或 该标签本身 """
    items = str(tag).split(CONNECTOR)
    if len(items) != 2:
        if get_none:
            return None
        else:
            return tag
    tail = items[1]
    return tail
