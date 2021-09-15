import tensorflow as tf

from bert_theme.base.bert import optimization
from bert_theme.base.helper.log_helper import get_logger
from bert_theme.base.model.models import create_model

logger = get_logger()


def file_based_input_fn(input_file, seq_length, num_themes, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "theme_id": tf.FixedLenFeature([num_themes], tf.int64)
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
            num_parallel_calls=4,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
            drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def model_fn_builder(
        bert_config, init_checkpoint, num_themes,
        learning_rate, num_train_steps, num_warmup_steps, args):
    """
    构建模型
    :param num_themes:
    :param args: 声明参数
    :param bert_config: bert模型配置信息
    :param init_checkpoint: 训练检查点采用此参数进行初始化
    :param learning_rate: 学习率
    :param num_train_steps: 总的训练步数
    :param num_warmup_steps: 热身训练步数
    :return: 返回 model_fn
    """

    def model_fn(features, labels, mode, params):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        theme_id = features["theme_id"]  # [batch_size, num_themes]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 构建模型
        loss, pred_dict, logits = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            themes=theme_id,
            num_themes=num_themes,
            dropout_rate=args.dropout_rate)

        from ..helper.ckpt_helper import CheckpointHelper

        checkpoint_helper = CheckpointHelper(args.output_dir, args.init_checkpoint)
        checkpoint_helper.load_model()

        # 训练
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = {
                'train_loss': loss,
                'predictions': pred_dict['pred_theme'],
                'labels': theme_id,
                'global_steps': tf.train.get_or_create_global_step(),
            }
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook])
        # 评估
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(_logits, _labels):
                eval_loss = tf.metrics.accuracy(labels=_labels, predictions=_logits)
                # eval_loss = tf.reduce_mean(
                #     tf.nn.softmax_cross_entropy_with_logits(
                #         logits=_logits, labels=_labels))
                return {"eval_loss": eval_loss}

            eval_metrics = metric_fn(logits, theme_id)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics
            )
        # 预测
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_dict)

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
