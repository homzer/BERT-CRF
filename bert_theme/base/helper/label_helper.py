import codecs
import os
import pickle


def init_label(output_dir):
    """
     读取 label 文件
    :param output_dir: 模型输出路径，搜寻 label2id.pkl 文件
    :return: num_labels 标签数量, id2label 字典
    """
    num_labels = 0
    with codecs.open(os.path.join(output_dir, 'theme2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = dict()
        for (key, value) in label2id.items():
            id2label[value] = key
            num_labels += 1
    return num_labels, id2label
