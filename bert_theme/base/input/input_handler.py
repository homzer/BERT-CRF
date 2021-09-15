import os

from bert_cause.base.helper.log_helper import get_logger
from bert_theme.base.input.input_helper import read_file, read_json_content
from bert_theme.base.input.input_example import create_example, create_example_for_server

logger = get_logger()


class InputHandler(object):
    """ 输入处理类 """

    def __init__(self, data_dir=None):
        self.data_dir = data_dir
        self.themes = ['非金融', '疫情金融', '金融', '疫情非金融']

    def get_train_examples(self):
        train_file = os.path.join(self.data_dir, "train.txt")
        relation_lines, themes = read_file(train_file)
        return create_example(relation_lines, themes, "train")

    def get_eval_examples(self):
        eval_file = os.path.join(self.data_dir, "eval.txt")
        relation_lines, themes = read_file(eval_file)
        return create_example(relation_lines, themes, "eval")

    def get_test_examples(self):
        test_file = os.path.join(self.data_dir, "test.txt")
        relation_lines, themes = read_file(test_file)
        return create_example(relation_lines, themes, "test")

    @staticmethod
    def get_pred_examples(content: list):
        """
        :param content: json 列表
        :return: InputExample 实例数组
        """
        logger.info(content)
        relation_texts, json_lines = read_json_content(content)
        return create_example_for_server(relation_texts, json_lines, "pred")

    def get_themes(self):
        return self.themes
