# -*- coding: UTF-8 -*-
import os

from bert_cause.base.bert import tokenization
from bert_cause.base.model.bert import modeling
from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.model import model_feature
from bert_cause.base.model import model_builder as builder
import tensorflow as tf

logger = get_logger()


def _get_bert_config(config_file):
    """
    获取bert模型的配置信息
    :param config_file: json配置文件路径名
    :return: 配置信息字典
    """
    return modeling.BertConfig.from_json_file(config_file)


def _get_run_config(model_dir, save_checkpoints_steps):
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True
    )

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=save_checkpoints_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        log_step_count_steps=save_checkpoints_steps,
        session_config=session_config
    )

    return run_config


class ModelHandler(object):
    def __init__(self, args, label_list):
        self.args = args
        self.label_list = label_list
        self.num_labels = len(self.label_list) + 1  # 这里为什么要加1，因为0索引被保留了；
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.args.vocab_file,
            do_lower_case=self.args.do_lower_case)

    def get_estimator(self, num_train_steps=None, num_warmup_steps=None):
        """
        :param num_train_steps: 需要训练的步数，默认为None，即可以不进行训练
        :param num_warmup_steps: 需要热身的步数，默认为None，即可以不进行热身
        :return: 返回的Estimator可以控制模型的训练，预测，评估工作等。
        """
        bert_config = _get_bert_config(self.args.bert_config_file)
        run_config = _get_run_config(self.args.output_dir, self.args.save_checkpoints_steps)
        # 检查最大长度是否超长
        if self.args.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.args.max_seq_length, bert_config.max_position_embeddings))

        # model_fn 是一个函数，其定义了模型，训练，评测方法，tf 新的架构方法，通过定义model_fn 函数，定义模型，
        # 然后通过EstimatorAPI进行模型的其他工作，Estimator就可以控制模型的训练，预测，评估工作等。
        model_fn = builder.model_fn_builder(
            bert_config=bert_config,
            num_labels=self.num_labels,  # 要保证label2id.pkl文件中最大索引能够取得到
            init_checkpoint=self.args.init_checkpoint,
            learning_rate=self.args.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            args=self.args)

        params = {'batch_size': self.args.batch_size}

        return tf.estimator.Estimator(model_fn, params=params, config=run_config)

    def train(self, train_examples, eval_examples):
        num_train_steps = int(len(train_examples) * 1.0 / self.args.batch_size *
                              self.args.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.args.warmup_proportion)
        num_eval_steps = int(len(eval_examples) * 1.0 / self.args.batch_size)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", self.args.batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", self.args.batch_size)
        logger.info("  Num steps = %d", num_eval_steps)

        estimator = self.get_estimator(num_train_steps, num_warmup_steps)
        # 1. 将数据转化为features
        train_file = os.path.join(self.args.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            model_feature.write_features(
                train_examples,
                self.label_list,
                self.args.max_seq_length,
                self.tokenizer,
                train_file,
                self.args.output_dir)

        eval_file = os.path.join(self.args.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            model_feature.write_features(
                eval_examples,
                self.label_list,
                self.args.max_seq_length,
                self.tokenizer,
                eval_file,
                self.args.output_dir)

        # 2. 构建input_fn
        train_input_fn = builder.file_based_input_fn(
            input_file=train_file,
            seq_length=self.args.max_seq_length,
            is_training=True,
            drop_remainder=True,
            num_parallel_cells=self.args.num_parallel_cells
        )

        eval_input_fn = builder.file_based_input_fn(
            input_file=eval_file,
            seq_length=self.args.max_seq_length,
            is_training=False,
            drop_remainder=False,
            num_parallel_cells=self.args.num_parallel_cells,
        )
        # 3. 构建钩子函数
        early_stopping_hook = builder.build_decrease_hook(
            estimator=estimator,
            num_train_steps=num_train_steps,
            save_checkpoints_steps=self.args.save_checkpoints_steps)

        # 4. 开始训练
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=num_train_steps,
            hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def predict(self, predict_examples):
        """ 预测并返回标签列表 [num_lines, max_seq_length]"""
        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", self.args.batch_size)

        predict_file = os.path.join(self.args.output_dir, "predict.tf_record")
        model_feature.write_features(
            predict_examples,
            self.label_list,
            self.args.max_seq_length,
            self.tokenizer,
            predict_file,
            self.args.output_dir)
        predict_input_fn = builder.file_based_input_fn(
            input_file=predict_file,
            seq_length=self.args.max_seq_length,
            is_training=False,
            drop_remainder=False,
            num_parallel_cells=self.args.num_parallel_cells
        )
        estimator = self.get_estimator()
        result = estimator.predict(input_fn=predict_input_fn)
        pred_ids = []
        for prediction in result:
            pred_ids.append(prediction["pred_ids"])
        return pred_ids
