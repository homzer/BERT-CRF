# -*- coding: UTF-8 -*-
import codecs
import os
import pickle


class ConstantLabel:
    CUN = 'CUN'
    CUS = 'CUS'
    CEN = 'CEN'
    EFN = 'EFN'
    EFS = 'EFS'
    EFCUN = 'EFCUN'
    EFCUS = 'EFCUS'

    Q_CUN = '原因中的核心名词'
    Q_CUS = '原因中的谓语或状态'
    Q_CEN = '中心词'
    Q_EFN = '结果中的核心名词'
    Q_EFS = '结果中的谓语或状态'

    B_CUN = 'B-CUN'
    I_CUN = 'I-CUN'
    B_CUS = 'B-CUS'
    I_CUS = 'I-CUS'
    B_CEN = 'B-CEN'
    I_CEN = 'I-CEN'
    B_EFN = 'B-EFN'
    I_EFN = 'I-EFN'
    B_EFS = 'B-EFS'
    I_EFS = 'I-EFS'
    B_EFCUN = 'B-EFCUN'
    I_EFCUN = 'I-EFCUN'
    B_EFCUS = 'B-EFCUS'
    I_EFCUS = 'I-EFCUS'

    O = 'O'
    B = 'B'
    I = 'I'
    E = 'E'
    S = 'S'
    X = 'X'

    CONNECTOR = '-'


def init_label(output_dir):
    """
     读取 label 文件
    :param output_dir: 模型输出路径，搜寻 label2id.pkl 文件
    :return: num_labels 标签数量, id2label 字典
    """
    num_labels = 0
    file = os.path.join(output_dir, 'label2id.pkl')
    with codecs.open(file, 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = dict()
        for (key, value) in label2id.items():
            id2label[value] = key
            num_labels += 1
    return num_labels, id2label
