# -*- coding: UTF-8 -*-
import os

from bert_cause.base.helper.label_helper import ConstantLabel as ConLabel
from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.input.input_example import create_example
from bert_cause.base.input.input_helper import read_file, read_text

logger = get_logger()

B_CUN = ConLabel.B_CUN
B_CUS = ConLabel.B_CUS
B_CEN = ConLabel.B_CEN
B_EFN = ConLabel.B_EFN
B_EFS = ConLabel.B_EFS
B_EFCUN = ConLabel.B_EFCUN
B_EFCUS = ConLabel.B_EFCUS
I_CUN = ConLabel.I_CUN
I_CUS = ConLabel.I_CUS
I_CEN = ConLabel.I_CEN
I_EFN = ConLabel.I_EFN
I_EFS = ConLabel.I_EFS
I_EFCUN = ConLabel.I_EFCUN
I_EFCUS = ConLabel.I_EFCUS

B = ConLabel.B
I = ConLabel.I
E = ConLabel.E
S = ConLabel.S
O = ConLabel.O


class InputHandler(object):
    """ 输入处理类 """

    def __init__(self, data_dir=None, max_seq_length=None, tokenizer=None):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.labels = [O, B_CUN, I_CUN, B_CUS, I_CUS, B_CEN, I_CEN, B_EFN,
                       I_EFN, B_EFS, I_EFS, B_EFCUN, I_EFCUN, B_EFCUS, I_EFCUS]
        self.tokenizer = tokenizer

    def get_train_examples(self):
        train_file = os.path.join(self.data_dir, "train.txt")
        content_texts, content_labels = read_file(train_file, self.max_seq_length)
        return create_example(content_texts=content_texts,
                              content_labels=content_labels,
                              set_type="train")

    def get_eval_examples(self):
        eval_file = os.path.join(self.data_dir, "eval.txt")
        content_texts, content_labels = read_file(eval_file, self.max_seq_length)
        return create_example(content_texts=content_texts,
                              content_labels=content_labels,
                              set_type="eval")

    def get_test_examples(self):
        test_file = os.path.join(self.data_dir, "test.txt")
        content_texts, content_labels = read_file(test_file, self.max_seq_length)
        return create_example(content_texts=content_texts,
                              content_labels=content_labels,
                              set_type="test")

    def get_pred_examples(self, content):
        content_texts = read_text(content, self.max_seq_length)
        return create_example(content_texts=content_texts,
                              content_labels=None,
                              set_type="pred")

    def get_labels(self):
        return self.labels
