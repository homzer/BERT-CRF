import codecs
import json

from bert_theme.base.bert import tokenization
from .input_example import InputExample

DOCUMENT = 'document'
TEXT = 'text'
QAS = 'qas'
THEME = 'label'
OTHER = '非金融'


def read_file(input_file):
    """Reads a Json data."""
    with codecs.open(input_file, 'r', encoding='utf-8') as file:
        relation_texts = []
        relation_themes = []
        for json_line in file:
            texts, themes = read_json(json_line)
            for text, theme in zip(texts, themes):
                relation_texts.append(text)
                relation_themes.append(theme)
    return relation_texts, relation_themes


def read_json_content(json_content: list) -> ([str], [dict]):
    """
    :param json_content: json列表
    :return:
    """
    relation_texts = []
    json_lines = []
    for json_line in json_content:
        texts, _ = read_json(json_line)
        for text in texts:
            relation_texts.append(text)
            json_lines.append(json_line)
    return relation_texts, json_lines


def read_json(json_line):
    """
    :param json_line: 一条json
    :return: relation_texts 浓缩因果关系列表，
    如果themed为 True 返回 relation themes主题列表
    """
    themes = []
    texts = []
    if type(json_line) is str:
        try:
            json_dict = json.loads(json_line)  # 将 json 格式的字符串转化为字典
        except json.decoder.JSONDecodeError:
            print(json_line)
            return texts, themes
    elif type(json_line) is dict:
        json_dict = json_line
    else:
        raise TypeError('json_line must be str or dict!')
    if QAS in json_dict.keys():
        qas = json_dict[QAS]
        for qa in qas:
            themes.append(get_theme(qa))
            texts.append(get_relation_text(qa))
    elif 'text' in json_dict.keys():
        texts.append(json_dict['text'])
        themes.append(OTHER)
    return texts, themes


def get_relation_text(qa):
    """ 将 qa 中的关系项按：原因名词-原因状态-中心词-结果名词-结果状态，拼接成字符串 """
    causes = []
    cause_status = []
    center = []
    effects = []
    effect_status = []

    def get_text_list(_list):
        texts = []
        for elem in _list:
            texts.append(elem.get('text', ''))
        return texts

    for item in qa:
        if 'question' not in item.keys() or 'answers' not in item.keys():
            continue
        question = item['question']
        answers = item['answers']
        if question == '原因中的核心名词':
            causes = get_text_list(answers)
        if question == '原因中的谓语或状态':
            cause_status = get_text_list(answers)
        if question == '中心词':
            center = get_text_list(answers)
        if question == '结果中的核心名词':
            effects = get_text_list(answers)
        if question == '结果中的谓语或状态':
            effect_status = get_text_list(answers)
    result = causes
    result.extend(cause_status)
    result.extend(center)
    result.extend(effects)
    result.extend(effect_status)
    return ''.join(result)


def get_theme(dict_list):
    """
     获取主题
    :param dict_list: 包含label键的字典列表
    :return: 其他，疫情，金融之中的一种
    """
    theme = OTHER
    for _dict in dict_list:
        if THEME in _dict.keys():
            theme = _dict.get(THEME)
            break
    return theme
