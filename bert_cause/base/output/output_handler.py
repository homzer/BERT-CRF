# -*- coding: UTF-8 -*-
import codecs
import os

from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.output import output_refine
from bert_cause.base.output.relation_group import create_groups_from_relation, create_json_dict
from bert_cause.base.output.relation_item import RelationItem

logger = get_logger()


class OutputHandler(object):

    def __init__(self, content_tokens, content_predicts, content_labels=None):
        """
        初始化输出处理类
        :param content_tokens: 预测文本
        :param content_predicts: 预测标签
        :param content_labels: 文本原始标签，可谓空
        """
        self.content_tokens = content_tokens
        self.content_labels = content_labels
        self.content_predicts = content_predicts
        assert len(self.content_tokens) == len(self.content_predicts)

    def optimizing_predict(self):
        """
         按照分词规则，进行标签优化
        """
        self.content_predicts = output_refine.optimize_predicts(self.content_tokens, self.content_predicts)

    def result_to_pair(self, output_dir):
        if self.content_labels is None:
            logger.info('Can not write to pair when content labels is None.')
            return
        output_file = os.path.join(output_dir, "label_test.txt")
        with codecs.open(output_file, 'w', encoding='utf-8') as writer:
            for line_tokens, line_labels, predict_labels in zip(
                    self.content_tokens,
                    self.content_labels,
                    self.content_predicts,
            ):
                predict_labels = predict_labels[0: len(line_tokens)]
                for j, label in enumerate(predict_labels):
                    try:
                        line = line_tokens[j] + ' ' + \
                               line_labels[j] + ' ' + label
                    except Exception as exc:
                        logger.info(exc)
                        logger.info(line_tokens)
                        logger.info(line_labels)
                        break
                    writer.write(line + '\n')
                writer.write('\n')

        # 结果评分分析
        from bert_cause.base.output.output_evaluate import Evaluator
        evaluator = Evaluator(output_file)
        evaluator.print_to_console()

    def result_to_relation(self, output_dir):
        relations_file = os.path.join(output_dir, "relations.txt")
        with codecs.open(relations_file, 'w', encoding='utf-8') as items_writer:
            for line_tokens, predict_labels in zip(self.content_tokens, self.content_predicts):
                item_handler = RelationItem()
                relation_items = item_handler.extract_items(line_tokens, predict_labels)
                relation_groups = create_groups_from_relation(relation_items)
                items_writer.write(str(relation_items) + '\n')
                items_writer.write('切片后：\n')
                for group in relation_groups:
                    items_writer.write(str(group) + '\n')
                items_writer.write('\n')

    def result_to_json(self):
        """ 将结果转化为 json 数组 """
        json_results = []
        for line_tokens, predict_labels in zip(self.content_tokens, self.content_predicts):
            item_handler = RelationItem()
            relation_items = item_handler.extract_items(line_tokens, predict_labels)
            relation_groups = create_groups_from_relation(relation_items)
            json_result = create_json_dict(relation_groups, line_tokens)
            json_results.append(json_result)
        return json_results

