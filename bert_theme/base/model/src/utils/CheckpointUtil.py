import collections
import re

import tensorflow as tf


def filter_compatible_params(checkpoint_file):
    """
    Filter the same parameters both in `trainable params` and `checkpoint params`.
    For example:
        in trainable params:
        <tf.Variable 'A:0' shape=(21128, 768) dtype=float32_ref>
        <tf.Variable 'B:0' shape=(512, 768) dtype=float32_ref>
        <tf.Variable 'C:0' shape=(768,) dtype=float32_ref>
        in checkpoint params:
        ('F', [21128, 768])
        ('B', [512, 768])
        ('D', [512, 768])
        ('C', [768])
        result:
        [<tf.Variable 'B:0' shape=(512, 768) dtype=float32_ref>,
        <tf.Variable 'C:0' shape=(768,) dtype=float32_ref>]
        Because only `C` and `B` are both shared by `trainable params` and `checkpoint params`
    :param checkpoint_file: directory of checkpoint file.
    :return: list of tf.Variable
    """
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    ckpt_vars = reader.get_variable_to_shape_map()
    ckpt_vars = sorted(ckpt_vars.items(), key=lambda x: x[0])
    train_vars = tf.trainable_variables()
    compatible_vars = []
    for ckpt_var in ckpt_vars:
        if type(ckpt_var) is str:
            ckpt_var_name = ckpt_var
        elif type(ckpt_var) is tuple:
            ckpt_var_name = ckpt_var[0]
        else:
            raise ValueError("Unknown checkpoint type: %s" % type(ckpt_var))
        for train_var in train_vars:
            train_var_name = re.match("^(.*):\\d+$", train_var.name).group(1)
            if train_var_name == ckpt_var_name:
                compatible_vars.append(train_var)
                break
    return compatible_vars


def print_param(name):
    """ Print the parameters in the graph. """
    with tf.Session() as sess:
        param = sess.graph.get_tensor_by_name(name)
        print(sess.run(param))


def print_checkpoint_variables(checkpoint_file):
    """ Print some variables from model checkpoint file. """
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    var_dict = reader.get_variable_to_shape_map()
    var_dict = sorted(var_dict.items(), key=lambda x: x[0])
    for item in var_dict:
        if 'adam' in item[0] or 'Adam' in item[0]:
            continue
        print(item)


def rename_checkpoint_variables(checkpoint_file, save_path="./result/model.ckpt"):
    """ Rename checkpoint variables """
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(checkpoint_file + '.meta')
    imported_meta.restore(sess, checkpoint_file)
    name_to_variable = collections.OrderedDict()
    for var in tf.global_variables():
        new_name = var.name
        if 'self/' in new_name:
            new_name = new_name.replace("self/", '')
        if 'bert/' in new_name:
            new_name = new_name.replace('bert/', '')
        if 'encoder' in new_name:
            new_name = new_name.replace('encoder', 'layer')
        if 'transformer' in new_name:
            new_name = new_name.replace('transformer', 'encoder')
        if ':0' in new_name:
            new_name = new_name.replace(':0', '')
        name_to_variable[new_name] = var
    saver = tf.train.Saver(name_to_variable)
    saver.save(sess, save_path)
