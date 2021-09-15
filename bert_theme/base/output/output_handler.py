import codecs
import json
import os

from bert_cause.base.helper.log_helper import get_logger

logger = get_logger()
QAS = "qas"


class OutputHandler(object):
    """ 输出处理类 """

    def __init__(self, relation_texts, predict_themes, relation_themes, json_dicts=None, relation_embeddings=None):
        self.relation_texts = relation_texts
        self.relation_themes = relation_themes
        self.predict_themes = predict_themes
        self.relation_embeddings = relation_embeddings
        self.json_dicts = json_dicts

    def result_to_pair(self, output_dir):
        relations_file = os.path.join(output_dir, "themes.txt")
        with codecs.open(relations_file, 'w', encoding='utf-8') as items_writer:
            for line_text, line_theme, pred_theme in zip(
                    self.relation_texts, self.relation_themes, self.predict_themes):
                items_writer.write("[样本标签：%s  预测标签：%s]\n" % (line_theme, pred_theme))
                items_writer.write(str(line_text))
                items_writer.write('\n\n')

    def result_to_npy(self):
        """ 输出序列的embeddings，以字典列表形式返回 """
        assert self.relation_embeddings is not None
        result = []
        for line_embeddings, line_text, line_theme in zip(
                self.relation_embeddings, self.relation_texts, self.predict_themes):
            item_dict = dict()
            item_dict['label'] = line_theme
            item_dict['text'] = line_text
            item_dict['embeddings'] = line_embeddings
            result.append(item_dict)
        return result

    def result_to_json(self) -> [dict]:
        """ 将主题标签插入原 json 中，返回 json 列表 """
        result = []
        for json_dict, theme in zip(self.json_dicts, self.predict_themes):
            assert type(json_dict) is dict
            if QAS in json_dict.keys():
                json_dict["qas"][0].append({"label": theme})
                result.append(json_dict)
            else:
                json_dict["label"] = theme
                result.append(json_dict)
        return result
