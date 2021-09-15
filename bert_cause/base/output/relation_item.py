# -*- coding: UTF-8 -*-
from bert_cause.base.output.output_refine import tag_tail
from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.helper.label_helper import ConstantLabel as ConLabel

logger = get_logger()

CUN = ConLabel.CUN
CUS = ConLabel.CUS
CEN = ConLabel.CEN
EFN = ConLabel.EFN
EFS = ConLabel.EFS
EFCUN = ConLabel.EFCUN
EFCUS = ConLabel.EFCUS
B = ConLabel.B
I = ConLabel.I
O = ConLabel.O
CONNECTOR = ConLabel.CONNECTOR


class RelationItem:
    """ 关系项处理类 """
    def __init__(self):
        self.items = []  # 关系项列表
        self.tail_set = {CEN, CUN, CUS, EFN, EFS, EFCUN, EFCUS}  # 合法的标注符尾

    def extract_items(self, tokens, labels):
        """
         将关系项提取出来
        :param tokens: 文本分字列表
        :param labels: 关系词标注列表
        :return: 关系项列表
        """
        if len(tokens) != len(labels):
            logger.info("Length of tokens %d != labels %d" % (len(tokens), len(labels)))
            logger.info(tokens)
            logger.info(labels)
            return self.items

        last_tag = O
        text = ''
        head_idx = None
        for idx, (token, label) in enumerate(zip(tokens, labels)):
            curr_tag = tag_tail(label, get_none=False)
            # jump up or down
            if last_tag != curr_tag:
                if last_tag in self.tail_set:
                    self.items_add(text, last_tag, head_idx)
                    head_idx = None
                if curr_tag in self.tail_set:
                    text = token
                    head_idx = idx
            # keep going
            if last_tag == curr_tag and last_tag in self.tail_set:
                text += token
            last_tag = curr_tag

        # add last item
        if last_tag in self.tail_set:
            self.items_add(text, last_tag, head_idx)

        self.post_work()

        return self.items

    def items_add(self, text, tag, head_idx):
        """
         将关系项添加进关系项列表中
        :param text: 关系项文本
        :param tag:  关系项类型 （如原因、结果、中心词）
        :param head_idx: 文本起始位置
        """
        self.items.append((text, tag, head_idx))

    def post_work(self):
        """
        后续工作，主要对 items 中的 EFCUN 和 EFCUS 做些调整 ,
        将 EFCUN EFCUN EFCUS 转化成：
        EFN EFN EFS CUN CUN CUS 的形式
        """

        def is_all_empty():
            """ 判断四个列表是否都为空 """
            length = len(effects) + len(effect_status) + len(causes) + len(cause_status)
            return length == 0

        def get_item_list():
            """ 将四个列表的内容合并成列表返回 """
            _list = []
            _list.extend(effects)
            _list.extend(effect_status)
            _list.extend(causes)
            _list.extend(cause_status)
            return _list

        effects = []
        effect_status = []
        causes = []
        cause_status = []
        new_items = []  # 用于替换之前的items
        for index, item in enumerate(self.items):
            tag = item[1]  # ('文本', '关系项类', 起始位置)
            if tag in [EFCUN, EFCUS]:
                if tag == EFCUN:
                    effects.append((item[0], EFN, item[2]))
                    causes.append((item[0], CUN, item[2]))
                if tag == EFCUS:
                    effect_status.append((item[0], EFS, item[2]))
                    cause_status.append((item[0], CUS, item[2]))
            elif is_all_empty():
                new_items.append(item)
            else:
                new_items.extend(get_item_list())  # 先添加先前的
                new_items.append(item)  # 再添加现在的
                # 将所有列表重置
                effects = []
                effect_status = []
                causes = []
                cause_status = []
        self.items = new_items
