# -*- coding: UTF-8 -*-
import os
import sys
from enum import Enum

from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.model.bert import modeling

logger = get_logger()
sys.path.append('..')

__all__ = ['PoolingStrategy', 'optimize_model']


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # corresponds to [CLS] for single sequences
    LAST_TOKEN = 5  # corresponds to [SEP] for single sequences
    CLS_TOKEN = 4  # corresponds to the first token for single seq.
    SEP_TOKEN = 5  # corresponds to the last token for single seq.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


def convert_variables_to_constants(sess,
                                   input_graph_def,
                                   output_node_names,
                                   variable_names_whitelist=None,
                                   variable_names_blacklist=None,
                                   use_fp16=False):
    from tensorflow.python.framework.graph_util_impl import extract_sub_graph
    from tensorflow.core.framework import graph_pb2
    from tensorflow.core.framework import node_def_pb2
    from tensorflow.core.framework import attr_value_pb2
    from tensorflow.core.framework import types_pb2
    from tensorflow.python.framework import tensor_util

    def patch_dtype(input_node, field_name, output_node):
        if use_fp16 and (field_name in input_node.attr) and (input_node.attr[field_name].type == types_pb2.DT_FLOAT):
            output_node.attr[field_name].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_HALF))

    inference_graph = extract_sub_graph(input_graph_def, output_node_names)

    variable_names = []
    variable_dict_names = []
    for node in inference_graph.node:
        if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
            variable_name = node.name
            if ((variable_names_whitelist is not None and
                 variable_name not in variable_names_whitelist) or
                    (variable_names_blacklist is not None and
                     variable_name in variable_names_blacklist)):
                continue
            variable_dict_names.append(variable_name)
            if node.op == "VarHandleOp":
                variable_names.append(variable_name + "/Read/ReadVariableOp:0")
            else:
                variable_names.append(variable_name + ":0")
    if variable_names:
        returned_variables = sess.run(variable_names)
    else:
        returned_variables = []
    found_variables = dict(zip(variable_dict_names, returned_variables))

    output_graph_def = graph_pb2.GraphDef()
    how_many_converted = 0
    for input_node in inference_graph.node:
        output_node = node_def_pb2.NodeDef()
        if input_node.name in found_variables:
            output_node.op = "Const"
            output_node.name = input_node.name
            dtype = input_node.attr["dtype"]
            data = found_variables[input_node.name]

            if use_fp16 and dtype.type == types_pb2.DT_FLOAT:
                output_node.attr["value"].CopyFrom(
                    attr_value_pb2.AttrValue(
                        tensor=tensor_util.make_tensor_proto(data.astype('float16'),
                                                             dtype=types_pb2.DT_HALF,
                                                             shape=data.shape)))
            else:
                output_node.attr["dtype"].CopyFrom(dtype)
                output_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(
                    tensor=tensor_util.make_tensor_proto(data, dtype=dtype.type,
                                                         shape=data.shape)))
            how_many_converted += 1
        elif input_node.op == "ReadVariableOp" and (input_node.input[0] in found_variables):
            # placeholder nodes
            # print('- %s | %s ' % (input_node.name, input_node.attr["dtype"]))
            output_node.op = "Identity"
            output_node.name = input_node.name
            output_node.input.extend([input_node.input[0]])
            output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
            if "_class" in input_node.attr:
                output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
        else:
            # mostly op nodes
            output_node.CopyFrom(input_node)

        patch_dtype(input_node, 'dtype', output_node)
        patch_dtype(input_node, 'T', output_node)
        patch_dtype(input_node, 'DstT', output_node)
        patch_dtype(input_node, 'SrcT', output_node)
        patch_dtype(input_node, 'Tparams', output_node)

        if use_fp16 and ('value' in output_node.attr) and (
                output_node.attr['value'].tensor.dtype == types_pb2.DT_FLOAT):
            # hard-coded value need to be converted as well
            output_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
                tensor=tensor_util.make_tensor_proto(
                    output_node.attr['value'].tensor.float_val[0],
                    dtype=types_pb2.DT_HALF)))

        output_graph_def.node.extend([output_node])

    output_graph_def.library.CopyFrom(inference_graph.library)
    return output_graph_def


def optimize_model(args, num_labels):
    """
    加载中文NER模型
    :param args:
    :param num_labels:
    :return:
    """
    try:
        pb_file = os.path.join(args.output_dir, 'bert_model.pb')
        if os.path.exists(pb_file):
            logger.info('pb_file exits: ' + str(pb_file))
            return pb_file

        import tensorflow as tf

        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                input_ids = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_ids')
                input_mask = tf.placeholder(tf.int32, (None, args.max_seq_len), 'input_mask')

                bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
                from bert_cause.base.model.models import create_model
                loss, prediction_dict = create_model(
                    bert_config=bert_config,
                    is_training=False,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    labels=None,
                    num_labels=num_labels,
                    use_one_hot_embeddings=False,
                    dropout_rate=1.0,
                    lstm_size=args.lstm_size)
                pred_ids = prediction_dict["pred_ids"]
                tf.identity(pred_ids, 'pred_ids')
                saver = tf.train.Saver()

            from bert_cause.base.model.model_builder import get_latest_checkpoint
            current_ckpt = get_latest_checkpoint(args.output_dir, args.init_checkpoint)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver.restore(sess, current_ckpt)
                logger.info('freeze...')
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_ids'])
                logger.info('model cut finished !!!')
        # 存储二进制模型到文件中
        logger.info('write graph to a tmp file: %s' % pb_file)
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        logger.error('fail to optimize the graph! %s' % e, exc_info=True)
