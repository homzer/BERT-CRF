# -*- coding: UTF-8 -*-
import os
import sys

from bert_cause.base.helper.log_helper import get_logger
from bert_theme.base.model.bert.bert_config import BertConfig

logger = get_logger()
sys.path.append('..')


def optimize_model(args, num_themes):
    """
    :param args:
    :param num_themes:
    :return:
    """
    try:
        pb_file = os.path.join(args.output_dir, 'bert_theme_model.pb')
        if os.path.exists(pb_file):
            logger.info('pb_file exits: ' + str(pb_file))
            return pb_file

        import tensorflow as tf

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')

                bert_config = BertConfig.from_json_file(args.bert_config_file)
                from bert_theme.base.model.models import create_model
                _, prediction_dict, _ = create_model(
                    bert_config=bert_config,
                    is_training=False,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    themes=None,
                    num_themes=num_themes)
                pred_theme = prediction_dict["pred_theme"]
                tf.identity(pred_theme, 'pred_theme')
                saver = tf.train.Saver()

            from bert_cause.base.model.model_builder import get_latest_checkpoint
            current_ckpt = get_latest_checkpoint(args.output_dir, args.init_checkpoint)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, current_ckpt)
                logger.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_theme'])
                logger.info('model cut finished !!!')
        # 存储二进制模型到文件中
        logger.info('write graph to a tmp file: %s' % pb_file)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        logger.error('fail to optimize the graph! %s' % e, exc_info=True)
