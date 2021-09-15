# -*- coding: UTF-8 -*-
from bert_cause.base.helper.label_helper import ConstantLabel as ConLabel
from bert_cause.base.helper.log_helper import get_logger

logger = get_logger()

CUN = ConLabel.CUN
CUS = ConLabel.CUS
CEN = ConLabel.CEN
EFN = ConLabel.EFN
EFS = ConLabel.EFS
EFCUN = ConLabel.EFCUN
EFCUS = ConLabel.EFCUS
B = ConLabel.B
I = ConLabel.I
O = ConLabel.O
CONNECTOR = ConLabel.CONNECTOR


def get_json_output(tokens, labels):
    """
    :param tokens: list 文本分词后的列表
    :param labels: list 标签列表
    :return: 关系项目列表
    """
    if len(tokens) != len(labels):
        raise ValueError('Length Not Equal!')
    # 获取一条文本的所有因果关系项
