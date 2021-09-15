import json


def get_from_predict_result(predict_result, id2theme):
    """ 获取所有预测标签 """
    predict_embeddings = []
    predict_themes = []
    for prediction in predict_result:
        predict_embeddings.append(prediction["embeddings"])
        predict_themes.append(id2theme[prediction["pred_theme"]])
    return predict_embeddings, predict_themes


def get_relation_texts(predict_examples):
    """
    获取因果关系对的文本
    :param predict_examples: InputExample 类的实例的列表
    :return: 文本列表
    """
    texts = []
    for predict_line in predict_examples:
        texts.append(str(predict_line.text))
    return texts


def get_json_dicts(predict_examples) -> [dict]:
    """
    获取因果关系对的json文本
    :param predict_examples: InputExample 类的实例的列表
    :return: json文本列表
    """
    json_dicts = []
    for example in predict_examples:
        json_dicts.append(json.loads(example.json_line))
    return json_dicts


def get_relation_themes(predict_examples):
    """
    获取所有关系对的主题
    :param predict_examples: InputExample 类的实例的列表
    :return: 主题列表
    """
    themes = []
    for example in predict_examples:
        themes.append(str(example.theme))
    return themes


def get_predict_themes(pred_ids, id2theme):
    """ 从预测结果中获取主题 """
    predict_themes = []
    for pred_id in pred_ids:
        predict_themes.append(id2theme[pred_id])
    return predict_themes
