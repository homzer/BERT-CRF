# -*- coding: UTF-8 -*-
import json

from bert_cause.base.helper.log_helper import get_logger

logger = get_logger()


def parse_json(content) -> (str, list):
    """
     从 json 中将文本和中心词取出，如果有多个因果关系对，则会包含多个中心词
    :param content: 一条 json 或者 dict
    :return: text文本，字符串类型
    centers中心词，(字符串, 起始位置)列表类型
    :raise JSONDecodeError 和 KeyError，该情况下返回的文本为 None, 数组为空
    """
    if type(content) == str:
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            logger.error("Json format is incorrect! %s " % content)
            return None, []
    elif type(content) == dict:
        pass
    else:
        logger.error("Expected type (dict) or (str) of content, got %s " % type(content))
        return None, []
    try:
        text = content["document"][0]["text"]
        qas = content["qas"]
        centers = []
        for qa in qas:
            for item in qa:
                if 'question' not in item:
                    print(item)
                    continue
                question = item['question']
                if question == '中心词':
                    centers.append((item['answers'][0]["text"], item['answers'][0]["start"]))
                    break
    except KeyError as e:
        logger.info(e)
        return None, []
    return text, centers


def load_jsons(content: list) -> list:
    """
     从文本数组中解析json
    :param content: 文本数组
    :return: 字典数组
    """
    result = []
    for line in content:
        if type(line) is dict:
            result.append(line)
        elif type(line) is str:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                result.append({'text': line})
    return result


def center_json(content: dict, centers: list) -> dict:
    """
    根据中心词提取相应因果关系对
    :param content: 包含'qas'键的字典
    :param centers: 中心词列表[(字符串，起始位置)]
    :return: content 更新'qas'后的字典
    """
    try:
        qas = content['qas']
        new_qas = []
        for qa in qas:
            for item in qa:
                question = item['question']
                if question == '中心词':
                    for center in centers:
                        if item['answers'][0]["text"] == center[0] and item['answers'][0]["start"] == center[1]:
                            new_qas.append(qa)
                            break
                    break
        content['qas'] = new_qas
    except KeyError as e:
        logger.info(e)
    return content
