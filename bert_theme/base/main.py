#! usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import pickle

import tensorflow as tf

from bert_theme.base.helper.ckpt_helper import get_last_checkpoint
from bert_theme.base.input.input_handler import InputHandler
from bert_theme.base.helper.log_helper import get_logger
from bert_theme.base.model.model_handler import ModelHandler
from bert_theme.base.output.output_handler import OutputHandler

# import

__version__ = '0.1.0'

logger = get_logger()


def adam_filter(model_path):
    """
    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的
    :param model_path:
    :return:
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))


def run_bert(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    # 在 re-train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if args.clean and args.do_train:
        clean_and_make_dir(args.output_dir)

    input_handler = InputHandler(args.data_dir)
    theme_list = input_handler.get_themes()
    model_handler = ModelHandler(args, theme_list)

    # 训练
    if args.do_train and args.do_eval:
        # 加载训练数据
        train_examples = input_handler.get_train_examples()
        eval_examples = input_handler.get_eval_examples()
        model_handler.train(train_examples, eval_examples)

    # 预测
    if args.do_predict:
        with codecs.open(os.path.join(args.output_dir, 'theme2id.pkl'), 'rb') as rf:
            theme2id = pickle.load(rf)
            id2theme = {value: key for key, value in theme2id.items()}
        # 获取输入参数与标签
        predict_examples = input_handler.get_test_examples()
        result = model_handler.predict(predict_examples)  # 获取预测结果值
        predict_themes = []
        predict_embeddings = []
        for prediction in result:
            predict_themes.append(id2theme[prediction['pred_theme']])
            predict_embeddings.append(prediction['embeddings'])

        # 结果输出处理
        import bert_theme.base.output.output_utils as utilities
        relation_texts = utilities.get_relation_texts(predict_examples)  # 获取关系文本列表
        relation_themes = utilities.get_relation_themes(predict_examples)  # 获取关系主题列表

        output_handler = OutputHandler(
            relation_texts=relation_texts,
            relation_themes=relation_themes,
            predict_themes=predict_themes,
            relation_embeddings=predict_embeddings)
        output_handler.result_to_pair(args.output_dir)
        output_handler.result_to_npy()

    # filter model
    if args.filter_adam_var:
        adam_filter(args.output_dir)


def clean_and_make_dir(output_dir):
    # 清除上一轮的输出
    if os.path.exists(output_dir):
        def del_file(path):
            ls = os.listdir(path)
            for i in ls:
                c_path = os.path.join(path, i)
                if os.path.isdir(c_path):
                    del_file(c_path)
                else:
                    os.remove(c_path)

        try:
            del_file(output_dir)
        except Exception as e:
            print(e)
            print('please remove the files of output dir and data.conf')
            exit(-1)

    # 创建新目录
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
