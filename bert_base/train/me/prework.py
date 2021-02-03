import os

import tensorflow as tf

from bert_base.bert import modeling
from bert_base.train.me import builder
from bert_base.train.me.processor import NerProcessor


def _get_bert_config(config_file):
    """
    获取bert模型的配置信息
    :param config_file: json配置文件路径名
    :return: 配置信息字典
    """
    return modeling.BertConfig.from_json_file(config_file)


def _get_run_config(model_dir):
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True
    )

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=100,
        save_checkpoints_steps=100,
        log_step_count_steps=100,
        session_config=session_config
    )

    return run_config


def _clean_last_output(output_dir):
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


def _make_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


def get_estimator(args, num_train_steps=None, num_warmup_steps=None):
    """
    :param args: 运行时的声明参数
    :param num_train_steps: 需要训练的步数，默认为None，即可以不进行训练
    :param num_warmup_steps: 需要热身的步数，默认为None，即可以不进行热身
    :return: 返回的Estimator可以控制模型的训练，预测，评估工作等。
    """
    bert_config = _get_bert_config(args.bert_config_file)
    # 检查最大长度是否超长
    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    processors = {
        "ner": NerProcessor
    }
    processor = processors[args.ner](args.output_dir)
    label_list = processor.get_labels()

    # model_fn 是一个函数，其定义了模型，训练，评测方法，tf 新的架构方法，通过定义model_fn 函数，定义模型，
    # 然后通过EstimatorAPI进行模型的其他工作，Estimator就可以控制模型的训练，预测，评估工作等。
    model_fn = builder.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        args=args)

    params = {
        'batch_size': args.batch_size
    }
    run_config = _get_run_config(args.output_dir)

    return tf.estimator.Estimator(model_fn, params=params, config=run_config)


def clean_and_make_output_dir(clean, do_train, output_dir):
    """
    在 re-train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    :param clean:
    :param do_train:
    :param output_dir:
    :return:
    """
    if clean and do_train:
        _clean_last_output(output_dir)
    _make_output_dir(output_dir)
