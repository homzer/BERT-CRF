# -*- coding: UTF-8 -*-
from bert_cause.base.helper.label_helper import ConstantLabel as ConLabel

CUN = ConLabel.CUN
CUS = ConLabel.CUS
CEN = ConLabel.CEN
EFN = ConLabel.EFN
EFS = ConLabel.EFS
TYPE_LIST = [CUN, CUS, CEN, EFS, EFN]
TYPE_DICT = {CUN: ConLabel.Q_CUN, CUS: ConLabel.Q_CUS, CEN: ConLabel.Q_CEN,
             EFN: ConLabel.Q_EFN, EFS: ConLabel.Q_EFS}


def create_groups_from_relation(relation_items):
    """
    从因果关系项创建关系组，一个因果关系项可能创建出多个关系组;
    因果关系项的顺序十分重要，应保持其为原文本的语义顺序，
    我们设想原因和原因状态谓语应集中在相邻区域，不考虑跨越幅度大的情况。
    :param relation_items: 因果关系项，类型是列表，
    存放一句文本的原因（结果）名词（谓语状态）、中心词；
    :return: 关系组列表
    """
    groups = RelationGroups()
    group = RelationGroup()
    for item in relation_items:
        if group.put(item):  # 尝试放入一个关系项
            continue
        else:  # 若放入失败则立马换下一个关系对
            groups.append(group)
            group = RelationGroup()
            group.put(item)
    groups.append(group)
    groups.self_adjust()
    return groups.groups_to_list()


def create_json_dict(relation_groups, line_tokens):
    """
    将因果关系组转化成 json 格式返回
    :param relation_groups: 因果关系组二维列表[[(文本，类型，起始位置), ]]
    :param line_tokens: 原文本列表
    :return: json 字典
    """
    result = {"document": [{"block_id": "0", "text": ''.join(line_tokens)}], "key": "", "status": 1, "qas": []}

    def add_to_qa(_item):
        text = _item[0]
        rel_type = _item[1]
        start = _item[2]
        if rel_type not in qa:
            qa[rel_type] = {"question": TYPE_DICT[rel_type], "answers": []}
        answer = {"start_block": "0", "start": start,
                  "end_block": 0, "end": start + len(text),
                  "text": text, "sub_answer": ""}
        qa[rel_type]["answers"].append(answer)

    def dict2list(_qa: dict) -> list:
        """ 将字典转化成列表 """
        return [value[1] for value in _qa.items()]

    qas = []
    for group in relation_groups:
        qa = {}
        for item in group:
            if len(item) != 3:
                print("Warning: Length of item != 3")
                continue
            add_to_qa(item)
        qas.append(dict2list(qa))
    result["qas"] = qas
    return result


class RelationGroup(object):
    def __init__(self):
        self.sequence = []
        self.causes = []
        self.effects = []
        self.center = []
        self.cause_status = []
        self.effect_status = []

    def put(self, item):
        """ 将一个关系项放入, 放入成功返回True, 否则False """
        if item is None:
            return False
        rel_type = item[1]  # 关系项类别
        if rel_type in TYPE_LIST:
            if self.is_needed(item):
                self.add(item)
                return True
        return False

    def is_needed(self, item):
        """ 检查一个关系项类别与上一个相同关系项类别的距离，来判断是否属于此关系对
         设想如果他们的距离过长，则很有可能是属于不同的关系对
         """
        rel_type = item[1]
        rel_start = item[2]
        if rel_type == CEN and len(self.center) > 0:  # 两个中心词肯定不要
            return False

        def type_distance(_list, _rel_type):
            """ 获取新加入项与最后一个相同项之间的距离，最小为1，找不到为0 """
            for i in range(len(_list)):
                index = len(_list) - i - 1
                if _list[index] == _rel_type:
                    return len(_list) - index
            return 0

        def text_distance(_rel_start):
            """ 计算文本上的距离，过远则不要 """
            if len(self.sequence) == 0:
                return 0
            last_type = self.sequence[-1]
            last_start = 0
            if last_type == CUN:
                last_start = self.causes[-1][2]
            elif last_type == CUS:
                last_start = self.cause_status[-1][2]
            elif last_type == CEN:
                last_start = self.center[-1][2]
            elif last_type == EFN:
                last_start = self.effects[-1][2]
            elif last_type == EFS:
                last_start = self.effect_status[-1][2]
            return abs(_rel_start - last_start)

        if type_distance(self.sequence, rel_type) <= 3:  # 距离很近，可以加入
            if text_distance(rel_start) <= 20:
                return True
        else:
            return False

    def add(self, item):
        """ 添加关系项 """
        if item is None:
            return
        rel_type = item[1]
        self.sequence.append(rel_type)
        if rel_type == CUN:
            self.causes.append(item)
        elif rel_type == CUS:
            self.cause_status.append(item)
        elif rel_type == CEN:
            self.center.append(item)
        elif rel_type == EFN:
            self.effects.append(item)
        elif rel_type == EFS:
            self.effect_status.append(item)

    def has_center(self):
        if len(self.center) == 1:
            return True
        else:
            return False

    def check_redundant(self):
        """ 检查冗余关系项 """
        redundant_items = set()
        if len(self.effects) >= 2:
            redundant_items.add(EFN)
        if len(self.effect_status) >= 2:
            redundant_items.add(EFS)
        if len(self.center) >= 2:  # unlikely to happen
            redundant_items.add(CEN)
        if len(self.causes) >= 2:
            redundant_items.add(CUN)
        if len(self.cause_status) >= 2:
            redundant_items.add(CUS)
        return redundant_items

    def check_insufficient(self):
        """ 检查不足关系项 """
        insufficient_items = set()
        if len(self.causes) == 0:
            insufficient_items.add(CUN)
        if len(self.cause_status) == 0:
            insufficient_items.add(CUS)
        if len(self.center) == 0:
            insufficient_items.add(CEN)
        if len(self.effect_status) == 0:
            insufficient_items.add(EFS)
        if len(self.effects) == 0:
            insufficient_items.add(EFN)
        return insufficient_items

    def has_redundant(self):
        return len(self.check_redundant()) >= 1

    def has_insufficient(self):
        return len(self.check_insufficient()) >= 1

    def query_for_item(self, item_type, from_tail=True):
        """
         请求 item_type 类型的关系项
        :param from_tail: 是否从 sequence 的尾部拿
        :param item_type: 属于 CUN, CUS, CEN, EFS, EFN 中的一种
        :return: 成功则返回关系项, 否则返回None
        """
        if item_type not in self.check_redundant():
            return None

        def remove_from_tail(_list, item):
            """ 从列表尾部开始搜索，删除第一个匹配到的item """
            result = []
            find_i = -1
            for i, elem in enumerate(_list):
                if elem == item:
                    find_i = i
            if find_i == -1:  # 未找到
                return _list
            else:
                for i, elem in enumerate(_list):
                    if i == find_i:
                        continue
                    result.append(elem)
            return result

        rel_item = None
        if from_tail:
            if item_type in self.sequence[len(self.sequence) - 2:]:
                # 从 sequence 中删除
                self.sequence = remove_from_tail(self.sequence, item_type)
                # 从关系项组中删除
                if item_type == CUN:
                    rel_item = self.causes.pop()
                elif item_type == CUS:
                    rel_item = self.cause_status.pop()
                elif item_type == CEN:  # 一般不可能出现
                    rel_item = self.center.pop()
                elif item_type == EFN:
                    rel_item = self.effects.pop()
                elif item_type == EFS:
                    rel_item = self.effect_status.pop()

        else:
            if item_type in self.sequence[0:2]:
                # 从 sequence 中删除
                self.sequence.remove(item_type)
                # 从关系项组中删除
                if item_type == CUN:
                    rel_item = self.causes[0]
                    self.causes = self.causes[1:]
                elif item_type == CUS:
                    rel_item = self.cause_status[0]
                    self.cause_status = self.cause_status[1:]
                elif item_type == CEN:  # 一般不可能出现
                    rel_item = self.center[0]
                    self.center = self.center[1:]
                elif item_type == EFN:
                    rel_item = self.effects[0]
                    self.effects = self.effects[1:]
                elif item_type == EFS:
                    rel_item = self.effect_status[0]
                    self.effect_status = self.effect_status[1:]
        return rel_item

    def query_for_cause(self, item_type):
        """
        获取原因名词或状态
        :return:
        """
        items = []
        if item_type == CUN:
            for cause in self.causes:
                items.append(cause)
        elif item_type == CUS:
            for cause_status in self.cause_status:
                items.append(cause_status)
        return items

    def to_tuple_lists(self):
        """ 以元组列表形式返回 """
        result = []
        result.extend(self.causes)
        result.extend(self.cause_status)
        result.extend(self.center)
        result.extend(self.effects)
        result.extend(self.effect_status)
        return result


class RelationGroups(object):
    def __init__(self):
        """ RelationGroup对象组管理类 """
        self.groups = []

    def append(self, group):
        if group.has_center():  # 只添加有中心词的关系组
            self.groups.append(group)

    def self_adjust(self):
        """ 组间进行调整关系项，以达到最优效果 """

        def get_last(groups):
            if i == 0:
                return None
            else:
                return groups[i - 1]

        def get_next(groups):
            if i == len(self.groups) - 1:
                return None
            else:
                return groups[i + 1]

        # 调整通常情况
        if len(self.groups) >= 2:
            for i, group in enumerate(self.groups):
                if group.has_insufficient():
                    last_group = get_last(self.groups)
                    if last_group is not None:
                        for item_type in group.check_insufficient():
                            group.add(last_group.query_for_item(item_type, from_tail=True))
                    next_group = get_next(self.groups)
                    if next_group is not None:
                        for item_type in group.check_insufficient():
                            group.add(next_group.query_for_item(item_type, from_tail=False))

            # 调整原因重叠情况
            for i, group in enumerate(self.groups):
                for item_type in group.check_insufficient():
                    if item_type in (CUN, CUS):
                        last_group = get_last(self.groups)
                        if last_group is not None:
                            for item in last_group.query_for_cause(item_type):
                                group.add(item)
                        if item_type in group.check_insufficient():
                            next_group = get_next(self.groups)
                            if next_group is not None:
                                for item in next_group.query_for_cause(item_type):
                                    group.add(item)

    def groups_to_list(self):
        result = []
        for group in self.groups:
            result.append(group.to_tuple_lists())
        return result
