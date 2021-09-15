# -*- coding: UTF-8 -*-
import tensorflow as tf
import os

from bert_cause.base.bert import optimization
from bert_cause.base.model.bert import modeling
from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.model.models import create_model

logger = get_logger()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def file_based_input_fn(input_file, seq_length, is_training, drop_remainder, num_parallel_cells=4):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, i_name_to_features):
        example = tf.parse_single_example(record, i_name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_calls=num_parallel_cells,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
            drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def model_fn_builder(
        bert_config, num_labels, init_checkpoint,
        learning_rate, num_train_steps, num_warmup_steps, args):
    """
    构建模型
    :param args: 声明参数
    :param bert_config: bert模型配置信息
    :param num_labels: 标签数量
    :param init_checkpoint: 训练检查点采用此参数进行初始化
    :param learning_rate: 学习率
    :param num_train_steps: 总的训练步数
    :param num_warmup_steps: 热身训练步数
    :return: 返回 model_fn
    """

    def model_fn(features, labels, mode, params):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)  # mode == 'train' 才训练

        # 使用参数构建bert_crf模型
        total_loss, pred_dict = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            labels=label_ids,
            num_labels=num_labels,
            use_one_hot_embeddings=False,
            dropout_rate=args.dropout_rate,
            lstm_size=args.lstm_size,
            cell=args.cell,
            num_layers=args.num_layers)

        tvars = tf.trainable_variables()
        # 加载预先训练过的模型
        current_ckpt = get_latest_checkpoint(args.output_dir, init_checkpoint)

        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, current_ckpt)
        tf.train.init_from_checkpoint(current_ckpt, assignment_map)

        # mode控制模型
        # 训练(train)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = {
                'train_loss': total_loss,
                'global_steps': tf.train.get_or_create_global_step(),
            }
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])
        # 评估(eval)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(_labels, _pred):
                a_loss = tf.metrics.mean_squared_error(labels=_labels, predictions=tf.cast(_pred, tf.int32))
                return {
                    "eval_loss": a_loss,
                }

            eval_metrics = metric_fn(
                label_ids,
                pred_dict['pred_ids']
            )
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        # 预测(infer)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_dict
            )

        return output_spec

    return model_fn


def build_decrease_hook(estimator, num_train_steps, save_checkpoints_steps):
    early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator=estimator,
        metric_name='loss',
        max_steps_without_decrease=num_train_steps,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=save_checkpoints_steps)
    return early_stopping_hook


def get_latest_checkpoint(output_dir, init_checkpoint):
    """ 在 output_dir 下找最新的检查点，若找不到则使用 init_checkpoint """
    # 加载预先训练过的模型
    latest_ckpt = tf.train.latest_checkpoint(output_dir)
    if latest_ckpt is None:
        # 使用初始化的检查点
        logger.info("Loading Trainable Variables From init_checkpoint: %s" % init_checkpoint)
        current_ckpt = init_checkpoint
    else:
        # 加载最新的检查点
        logger.info("Loading Trainable Variables From latest_checkpoint: %s" % latest_ckpt)
        current_ckpt = latest_ckpt
    return current_ckpt
