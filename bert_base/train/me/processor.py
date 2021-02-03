import codecs
import os
import pickle

from bert_base.bert import tokenization
from bert_base.train.models import InputExample


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join([label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines


class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(self._read_data(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        if labels is not None:
            try:
                # 支持从文件中读取标签类型
                if os.path.exists(labels) and os.path.isfile(labels):
                    with codecs.open(labels, 'r', encoding='utf-8') as fd:
                        for line in fd:
                            self.labels.append(line.strip())
                else:
                    # 否则通过传入的参数，按照逗号分割
                    self.labels = labels.split(',')
                self.labels = set(self.labels)  # to set
            except Exception as e:
                print(e)
        # 通过读取train文件获取标签的方法会出现一定的风险。
        if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as rf:
                self.labels = pickle.load(rf)
        else:
            if len(self.labels) > 0:
                self.labels = self.labels.union({"X", "[CLS]", "[SEP]"})
                with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
                    pickle.dump(self.labels, rf)
            else:
                self.labels = ["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X",
                               "[CLS]", "[SEP]"]
        return self.labels

    @staticmethod
    def _create_example(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples  # 取数据的一部分

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()  # 移除字符串首尾换行符和空格
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])  # 取切片倒数第一个字符
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                self.labels.add(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])  # ' '.join(list) 将 ['A','B','C'] 转化成 'A B C'
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):  # 是否以指定字符串开头
                    continue
            return lines
