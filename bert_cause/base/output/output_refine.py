# -*- coding: UTF-8 -*-
from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.helper.label_helper import ConstantLabel as ConLabel

logger = get_logger()
B = ConLabel.B
I = ConLabel.I
CONNECTOR = ConLabel.CONNECTOR
CEN = ConLabel.CEN


def tag_tail(tag, get_none=True):
    """ 返回一个标签的尾部，必须是 X-Y 的形式，否则返回 None 或 该标签本身 """
    items = str(tag).split(CONNECTOR)
    if len(items) != 2:
        if get_none:
            return None
        else:
            return tag
    tail = items[1]
    return tail


def optimize_predicts(context_tokens, context_predicts):
    """
    基于分词规则对预测的标签进行优化
    :param context_tokens: 二维列表 [lines_num, seq_length] 文本分词后的列表
    :param context_predicts: 二维列表 [lines_num, seq_length] 神经网络预测的标签
    :return: 优化后标签的二维列表 [lines_num, seq_length]
    """
    import jieba
    if len(context_predicts) != len(context_tokens):
        print(context_tokens)
        print(context_predicts)
        raise ValueError("The number of texts != labels! "
                         "got context_tokens %d and context_predicts %d" %
                         (len(context_tokens), len(context_predicts)))

    def merge_special_word():
        """ 将分词后的 '[', 'UNK', ']' 转化成 'O' """
        line = ' '.join(text_words)
        line = line.replace('[ UNK ]', 'O')
        return line.split(' ')

    opt_predicts = []
    for line_tokens, line_predicts in zip(context_tokens, context_predicts):
        line_predicts = line_predicts[0:len(line_tokens)]
        cut = jieba.cut(''.join(line_tokens))
        text_words = []
        for word in cut:
            text_words.append(word)

        text_words = merge_special_word()

        tagged_text = tag_entity(text_words)
        opt_predicts.append(full_fill(tagged_text, line_predicts))
    return opt_predicts


def tag_entity(words):
    tapped = []
    for word in words:
        for i, letter in enumerate(list(word)):
            if i == 0:
                tapped.append(B)
            else:
                tapped.append(I)
    return tapped


def full_fill(tag_std, tag_fill):
    tag_len = len(tag_fill)
    if len(tag_std) != tag_len:
        print(tag_std)
        print(tag_fill)
        raise ValueError('the length of standard tags not equal with the fill tags! '
                         'text: %d, label: %d' % (len(tag_std), len(tag_fill)))
    # backward search
    for i, curr_tap in enumerate(tag_fill):
        curr_tap_tail = tag_tail(curr_tap)
        if curr_tap_tail is None:
            continue
        if i + 1 >= tag_len:
            break
        next_tap = tag_fill[i + 1]
        next_tap_tail = tag_tail(next_tap)
        if curr_tap_tail != next_tap_tail:
            if curr_tap_tail == CEN:  # we suspect that 'CEN' do not need to expand in most case
                continue
            if tag_std[i + 1] == I:
                tag_fill[i + 1] = I + CONNECTOR + curr_tap_tail

    # forward search
    for j in range(tag_len):
        i = tag_len - j - 1
        curr_tap = tag_fill[i]
        curr_tap_tail = tag_tail(curr_tap)
        if curr_tap_tail is None:
            continue
        if i - 1 < 0:
            break
        next_tap = tag_fill[i - 1]
        next_tap_tail = tag_tail(next_tap)
        if curr_tap_tail != next_tap_tail:
            if tag_std[i] == I:
                tag_fill[i - 1] = B + CONNECTOR + curr_tap_tail
                tag_fill[i] = I + CONNECTOR + curr_tap_tail
                continue
            if tag_std[i] == B:
                tag_fill[i] = B + CONNECTOR + curr_tap_tail

    # assimilate
    for j in range(1, tag_len - 1):
        last_tap = tag_fill[j - 1]
        curr_tap = tag_fill[j]
        next_tap = tag_fill[j + 1]
        last_tail = tag_tail(last_tap)
        next_tail = tag_tail(next_tap)
        curr_tail = tag_tail(curr_tap)
        if last_tail == next_tail and curr_tail is not None and next_tail is not None:
            tag_fill[j] = I + CONNECTOR + next_tail

    return tag_fill
