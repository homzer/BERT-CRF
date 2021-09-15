# -*- coding: UTF-8 -*-
import tensorflow as tf


class CheckpointHelper(object):
    def __init__(self, ckpt_path):
        self.ckpt_file = ckpt_path

    def print_variables(self):
        reader = tf.train.NewCheckpointReader(self.ckpt_file)
        var_dict = reader.get_variable_to_shape_map()
        var_dict = sorted(var_dict.items(), key=lambda x: x[0])
        for item in var_dict:
            if 'adam_v' in item[0] or 'adam_m' in item[0]:
                continue
            print(item)

