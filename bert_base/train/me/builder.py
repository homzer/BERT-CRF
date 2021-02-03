import tensorflow as tf

from bert_base.bert import modeling, optimization
from bert_base.server.helper import set_logger
from bert_base.train.models import create_model

logger = set_logger('ME Builder')


def build_file_based_input_fn(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
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
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=4,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
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
        # logger.info("*** Features ***")
        # for name in sorted(features.keys()):
        #     logger.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)  # mode == 'train' 才训练

        # 使用参数构建bert_crf模型
        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)

        tvars = tf.trainable_variables()
        # 加载预先训练过的模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            logger.info("Loading Trainable Variables From %s" % init_checkpoint)

            # 打印变量名
            # logger.info("**** Loaded Trainable Variables ****")
            # 打印加载模型的参数
            # for var in tvars:
            #     init_string = ""
            #     if var.name in initialized_variable_names:
            #         init_string = ", *INIT_FROM_CHECKPOINT*"
            #     logger.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)

        # mode控制模型
        # 训练(train)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = {'loss': total_loss, 'global_steps': tf.train.get_or_create_global_step()}
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])
        # 评估(eval)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(i_label_ids, i_pred_ids):
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=i_label_ids, predictions=i_pred_ids),
                }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        # 预测(infer)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
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

