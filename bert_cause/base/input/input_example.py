# -*- coding: UTF-8 -*-
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


def create_example(content_texts, content_labels, set_type):
    """
    创建 InputFeatures
    :param content_labels: 标签二维列表
    :param content_texts: 文本列表
    :param set_type: 标识符号
    :return: InputFeature 列表
    """

    examples = []
    if content_labels is None:
        for (i, tokens) in enumerate(content_texts):
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, text=tokens))
        return examples

    for (i, (tokens, labels)) in enumerate(zip(content_texts, content_labels)):
        guid = "%s-%s" % (set_type, i)
        examples.append(InputExample(guid=guid, text=tokens, label=labels))
    return examples
