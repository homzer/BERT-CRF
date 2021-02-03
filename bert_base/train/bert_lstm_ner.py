#! usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import pickle

import tensorflow as tf

from bert_base.bert import tokenization
from bert_base.server.helper import set_logger
from bert_base.train.me import prework, convertion, builder, data_processor

# import

__version__ = '0.1.0'


logger = set_logger('NER Training')


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


def _do_train(estimator, train_input_fn, num_train_steps, early_stopping_hook, eval_input_fn):
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=num_train_steps,
        hooks=[early_stopping_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def run_bert(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

    prework.clean_and_make_output_dir(
        args.clean, args.do_train, args.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    # 加载训练数据
    if args.do_train and args.do_eval:
        train_examples, num_train_steps, num_warmup_steps, eval_examples = data_processor.load_data(args)

    label_list = data_processor.get_labels(args.output_dir)
    estimator = prework.get_estimator(args, num_train_steps, num_warmup_steps)

    if args.do_train and args.do_eval:
        # 1. 将数据转化为features
        train_file = os.path.join(args.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            convertion.filed_based_convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, train_file, args.output_dir)

        eval_file = os.path.join(args.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            convertion.filed_based_convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, eval_file, args.output_dir)

        # 2. 构建input_fn
        train_input_fn = builder.build_file_based_input_fn(
            input_file=train_file,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=True)

        eval_input_fn = builder.build_file_based_input_fn(
            input_file=eval_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # 3. 构建钩子函数
        early_stopping_hook = builder.build_decrease_hook(
            estimator=estimator,
            num_train_steps=num_train_steps,
            save_checkpoints_steps=args.save_checkpoints_steps)

        # 4. 开始训练
        _do_train(
            estimator=estimator,
            train_input_fn=train_input_fn,
            num_train_steps=num_train_steps,
            early_stopping_hook=early_stopping_hook,
            eval_input_fn=eval_input_fn)

    if args.do_predict:
        token_path = os.path.join(args.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        # 获取输入参数与标签
        predict_examples = data_processor.get_test_examples(args.output_dir, args.data_dir)
        predict_file = os.path.join(args.output_dir, "predict.tf_record")
        convertion.filed_based_convert_examples_to_features(predict_examples, label_list,
                                                            args.max_seq_length, tokenizer,
                                                            predict_file, args.output_dir, mode="test")

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", args.batch_size)

        predict_drop_remainder = False
        predict_input_fn = builder.build_file_based_input_fn(
            input_file=predict_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)  # 获取预测结果值
        output_predict_file = os.path.join(args.output_dir, "label_test.txt")

        def result_to_pair(i_writer):
            for predict_line, prediction in zip(predict_examples, result):
                idx = 0
                line = ''
                line_token = str(predict_line.text).split(' ')
                label_token = str(predict_line.label).split(' ')
                len_seq = len(label_token)
                if len(line_token) != len(label_token):
                    logger.info(predict_line.text)
                    logger.info(predict_line.label)
                    break
                for id in prediction:
                    if idx >= len_seq:
                        break
                    if id == 0:
                        continue
                    curr_labels = id2label[id]
                    if curr_labels in ['[CLS]', '[SEP]']:
                        continue
                    try:
                        line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                    except Exception as exc:
                        logger.info(exc)
                        logger.info(predict_line.text)
                        logger.info(predict_line.label)
                        line = ''
                        break
                    idx += 1
                i_writer.write(line + '\n')

        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)
        from bert_base.train import conlleval
        eval_result = conlleval.return_report(output_predict_file)
        print(''.join(eval_result))
        # 写结果到文件中
        with codecs.open(os.path.join(args.output_dir, 'predict_score.txt'), 'a', encoding='utf-8') as fd:
            fd.write(''.join(eval_result))
    # filter model
    if args.filter_adam_var:
        adam_filter(args.output_dir)
