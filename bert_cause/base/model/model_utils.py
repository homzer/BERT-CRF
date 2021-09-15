# -*- coding: UTF-8 -*-
def one_hot_encoding(class_ids, n_classes):
    """
    使用 one_hot 编码方式
    :param class_ids: [seq_length] 必须被具体赋值
    :param n_classes:
    :return: [seq_length, n_classes]
    """
    from keras.utils import np_utils
    class_ids = np_utils.to_categorical(class_ids, n_classes, dtype=int)
    return class_ids


def reshape_to_list(list_2d):
    """ 将二维列表转化为一维 """
    list_1d = []
    for item in list_2d:
        list_1d.extend(item)
    return list_1d
