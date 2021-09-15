import matplotlib.pyplot as plt
import numpy as np
from joblib.numpy_pickle_utils import xrange
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

__all__ = ['dimension_to_3d']


# 遍历embedding，并对数组进行余项归零和向量拼接方法
def pre_data_trans_array(dicts):
    all_embeddings = []
    for _dict in dicts:
        embeddings = _dict['embeddings']  # shape(128, 768)
        truth_length = len(_dict['text'])
        max_length = np.shape(embeddings)[0]

        padding = [1] * truth_length
        for i in range(max_length - truth_length):
            padding.append(0)
        padding = padding[:max_length]
        padding = np.expand_dims(padding, axis=-1)
        all_embeddings.append(np.reshape(np.multiply(padding, embeddings), newshape=[-1]))
    # 列表属性转换为数组
    array_all_embeddings = np.array(all_embeddings)
    return array_all_embeddings


# 数据归一化操作方法
def trans_normal(array):
    max_cols = array.max(axis=0)
    min_cols = array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    normal_array = np.empty((data_rows, data_cols))
    for i in xrange(data_cols):
        normal_array[:, i] = (array[:, i] - min_cols[i]) / (max_cols[i] - min_cols[i])
    return normal_array


# PCA降维操作，将原本726*512维度向量降维成3维向量
def reduce_dimension_pca(array_all_embeddings):
    digital_pca = PCA(n_components=3)
    pca_data = digital_pca.fit_transform(array_all_embeddings)
    return pca_data


# LDA降维操作，将原本726*512维度向量降维成3维向量
def reduce_dimension_lda(array_all_embeddings, label):
    digital_lda = LinearDiscriminantAnalysis(n_components=3)
    lda_data = digital_lda.fit_transform(array_all_embeddings, label)
    return lda_data


# MDS降维算法
def reduce_dimension_mds(array_all_embeddings):
    digital_mds = manifold.MDS(n_components=3)
    mds_data = digital_mds.fit_transform(array_all_embeddings)
    return mds_data


def data_3d_plot(array):
    x = array[:, 0]
    y = array[:, 1]
    z = array[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c='r', label='test')
    ax.legend(loc='best')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()


def dimension_to_3d(data_dicts):
    """
    将高维向量降维至三维
    :param data_dicts: 字典列表，字典格式为 {'label': str, 'text': str, 'embeddings': 2D array}
    :return: 字典列表，字典格式为 {'label': str, 'text': str, 'pos': [x, y ,z]}
    """
    data_array = pre_data_trans_array(data_dicts)

    after_mds_data = reduce_dimension_mds(data_array)
    after_normal_data_2 = trans_normal(after_mds_data)
    after_normal_data_2 = after_normal_data_2.tolist()

    result = []
    for data_dict, pos in zip(data_dicts, after_normal_data_2):
        item = {"label": data_dict['label'], "text": data_dict['text'], "pos": pos}
        result.append(item)
    return result

