#! usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import pickle

import tensorflow as tf

from bert_cause.base.input.input_handler import InputHandler
from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.model.model_handler import ModelHandler
from bert_cause.base.output.output_handler import OutputHandler

__version__ = '0.1.0'

logger = get_logger()


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

    from bert_cause.base.bert.tokenization import FullTokenizer
    tokenizer = FullTokenizer(args.vocab_file)
    input_handler = InputHandler(data_dir=args.data_dir, max_seq_length=args.max_seq_length, tokenizer=tokenizer)
    label_list = input_handler.get_labels()
    model_handler = ModelHandler(args, label_list)

    # 训练
    if args.do_train and args.do_eval:
        # 加载训练数据
        train_examples = input_handler.get_train_examples()
        eval_examples = input_handler.get_eval_examples()
        model_handler.train(train_examples, eval_examples)

    # 预测
    if args.do_predict:
        token_path = os.path.join(args.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)
        from bert_cause.base.helper.label_helper import init_label
        num_labels, id2label = init_label(args.output_dir)
        # 获取输入参数与标签
        predict_examples = input_handler.get_test_examples()
        pred_ids = model_handler.predict(predict_examples)  # 获取预测结果值

        # 结果输出处理
        import bert_cause.base.output.output_utils as utilities
        content_tokens = utilities.get_context_tokens(predict_examples)  # 获取句子分字列表
        content_predicts = utilities.get_predict_labels(pred_ids, id2label)  # 获取预测标签
        content_labels = utilities.get_context_labels(predict_examples)  # 获取所有标签列表
        output_handler = OutputHandler(
            content_tokens=content_tokens,
            content_predicts=content_predicts,
            content_labels=content_labels)
        # output_handler.optimizing_predict()
        output_handler.result_to_pair(args.output_dir)
        # output_handler.result_to_relation(args.output_dir)
        # output_handler.result_to_json()

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
