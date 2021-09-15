import json

from bert_theme.base.bert import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text, guid=None, theme=None, json_line=None):
        self.guid = guid
        self.text = text
        self.theme = theme
        self.json_line = json_line


def create_example(lines, themes, set_type):
    examples = []
    for (i, (line, theme)) in enumerate(zip(lines, themes)):
        guid = "%s-%s" % (set_type, i)
        text = tokenization.convert_to_unicode(line)
        theme = tokenization.convert_to_unicode(theme)
        if len(text) == 0:
            continue
        examples.append(InputExample(guid=guid, text=text, theme=theme))
    return examples


def create_example_for_server(texts, json_lines, set_type):
    examples = []
    for (i, (text, json_line)) in enumerate(zip(texts, json_lines)):
        guid = "%s-%s" % (set_type, i)
        text = tokenization.convert_to_unicode(text)
        json_line = tokenization.convert_to_unicode(json.dumps(json_line, ensure_ascii=False))
        examples.append(InputExample(guid=guid, text=text, json_line=json_line))
    return examples
