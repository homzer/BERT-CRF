import codecs
import collections
import os
import re

import tensorflow as tf

from bert_theme.base.helper.log_helper import get_logger

logger = get_logger()


class CheckpointHelper(object):
    def __init__(self, output_dir, ckpt_file):
        self.output_dir = output_dir
        self.ckpt_file = ckpt_file

    def print_variables(self):
        reader = tf.train.NewCheckpointReader(self.ckpt_file)
        var_dict = reader.get_variable_to_shape_map()
        var_dict = sorted(var_dict.items(), key=lambda x: x[0])
        for item in var_dict:
            if 'adam_v' in item[0] or 'adam_m' in item[0]:
                continue
            print(item)

    def load_model(self):
        """ 根据检查点文件初始化模型参数 """
        tvars = tf.trainable_variables()
        # 加载预先训练过的模型
        latest_ckpt = tf.train.latest_checkpoint(self.output_dir)
        if latest_ckpt is None:
            # 使用初始化的检查点
            logger.info("Loading Trainable Variables From init_checkpoint: %s" % self.ckpt_file)
            current_ckpt = self.ckpt_file
        else:
            # 加载最新的检查点
            logger.info("Loading Trainable Variables From latest_checkpoint: %s" % latest_ckpt)
            current_ckpt = latest_ckpt
        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, current_ckpt)
        tf.train.init_from_checkpoint(current_ckpt, assignment_map)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        logger.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last
