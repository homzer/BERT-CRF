# -*- coding: UTF-8 -*-
import codecs
from collections import OrderedDict

from bert_cause.base.helper.tag_helper import parse_tag


class EvalCount(object):
    def __init__(self, tag_type):
        self.tag_type = tag_type
        self.tp = 0
        self.fp = 0
        self.positives = 0

    def positive_gain(self):
        self.positives += 1

    def true_positive_gain(self):
        self.tp += 1

    def false_positive_gain(self):
        self.fp += 1

    def precision(self):
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp * 1.0 / (self.tp + self.fp)

    def recall(self):
        if self.positives == 0:
            return 0.0
        return self.tp * 1.0 / self.positives

    def found(self):
        return self.positives


class Evaluator:
    def __init__(self, input_file):
        self.input_file = input_file
        self.all_count = 0
        self.same_count = 0
        self.eval_dict = OrderedDict()
        self.evaluate()

    def get_eval_count(self, key):
        """ 获取字典中 EvalCount 对象 """
        if key not in self.eval_dict:
            self.eval_dict[key] = EvalCount(key)
        return self.eval_dict[key]

    def add_positives(self, correct):
        """ 为当前类别 correct 添加一个正项 """
        count = self.get_eval_count(correct)
        count.positive_gain()

    def eval_check(self, correct, guessed):
        """ 统计当前 guessed 是否为 correct 的真正项还是假正项 """
        if correct == guessed:
            count = self.get_eval_count(correct)
            count.true_positive_gain()
        else:
            count = self.get_eval_count(guessed)
            count.false_positive_gain()

    def evaluate(self):
        with codecs.open(self.input_file, "r", "utf8") as context:
            for line in context:
                line = line.rstrip('\r\n')
                features = line.split()
                if len(features) < 3:
                    continue
                guessed = features[2]
                correct = features[1]
                self.add_positives(correct)
                self.eval_check(correct, guessed)
                self.all_count += 1
                if guessed == correct:
                    self.same_count += 1

    def print_to_console(self):
        result = OrderedDict()
        for (key, item) in self.eval_dict.items():
            _, tag = parse_tag(key)
            if len(tag) == 0:
                continue
            if tag not in result:
                result[tag] = [0.0, 0.0, 0]  # precision, recall, found
            _list = result[tag]
            found = _list[2] + item.found()
            if found == 0:
                continue
            portion_1 = _list[2] * 1. / found
            portion_2 = item.found() * 1. / found
            precision = _list[0] * portion_1 + item.precision() * portion_2
            recall = _list[1] * portion_1 + item.recall() * portion_2
            new_list = [precision, recall, found]
            result[tag] = new_list

        total_precision = 0.0
        total_recall = 0.0
        total_found = 0
        for (key, item) in result.items():
            line = '%-8s precision: %6.2f%%; recall: %6.2f%%; found: %4d' \
                   % (key, item[0] * 100., item[1] * 100., item[2])
            total_found += item[2]
            total_precision += item[0] * item[2]
            total_recall += item[1] * item[2]
            print(line)
        total_precision /= total_found
        total_recall /= total_found
        total_f1 = 2. * total_recall * total_precision / (total_recall + total_precision)
        total_accuracy = self.same_count * 1. / self.all_count if self.all_count != 0 else 0.
        print('accuracy: %6.2f%%; precision: %6.2f%%; recall: %6.2f%%; f1: %4.2f' %
              (total_accuracy * 100., total_precision * 100., total_recall * 100., total_f1))


if __name__ == '__main__':
    output_file = "../../result/label_test.txt"
    evaluator = Evaluator(output_file)
    evaluator.print_to_console()
