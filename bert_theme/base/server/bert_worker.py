# -*- coding: UTF-8 -*-
import multiprocessing
import os
from multiprocessing import Process

import zmq
import zmq.decorators as zmqd
from zmq.utils import jsonapi

from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.server.server_helper import send_ndarray
from bert_cause.base.server.zmq_decor import multi_socket

logger = get_logger()


class BertWorker(Process):
    def __init__(self, id, args, worker_address_list, sink_address, device_id, graph_path, id2theme=None):
        Process.__init__(self)
        self.worker_id = id
        self.device_id = device_id
        self.max_seq_len = args.max_seq_len
        self.mask_cls_sep = args.mask_cls_sep
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.verbose = args.verbose
        self.graph_path = graph_path
        self.use_fp16 = args.fp16
        self.vocab_file = args.vocab_file
        self.id2theme = id2theme
        self.examples = []

    def close(self):
        logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        logger.info('terminated!')

    def get_estimator(self, tf):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            input_map = {"input_ids": input_ids, "input_mask": input_mask}
            pred_theme, embeddings = tf.import_graph_def(
                graph_def, name='', input_map=input_map,
                return_elements=['theme/logits/ArgMax:0', 'bert/transformer/Reshape_2:0'])

            return EstimatorSpec(mode=mode, predictions={
                'client_id': features['client_id'],
                'pred_themes': pred_theme,
                'embeddings': embeddings
            })

        # 0 表示只使用CPU 1 表示使用GPU
        config = tf.ConfigProto(device_count={'GPU': 0 if self.device_id < 0 else 1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config))

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink, *receivers):

        logger.info('use device %s, load graph from %s' %
                    ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), self.graph_path))

        tf = import_tf(self.device_id, self.verbose, use_fp16=self.use_fp16)
        estimator = self.get_estimator(tf)
        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink.connect(self.sink_address)

        import bert_theme.base.output.output_utils as utilities
        from bert_theme.base.output.output_handler import OutputHandler
        from bert_theme.base.reduce.reduce_dimension import dimension_to_3d
        for result in estimator.predict(input_fn=self.input_fn_builder(receivers, tf), yield_single_examples=False):
            relation_texts = utilities.get_relation_texts(self.examples)  # 获取句子分字列表
            predict_themes = utilities.get_predict_themes(result['pred_themes'], self.id2theme)  # 获取预测标签
            json_dicts = utilities.get_json_dicts(self.examples)  # 获取json字典列表
            assert type(json_dicts[0]) is dict
            relation_embeddings = result['embeddings']
            logger.info(json_dicts)
            output_handler = OutputHandler(
                relation_texts=relation_texts,
                predict_themes=predict_themes,
                relation_embeddings=relation_embeddings,
                json_dicts=json_dicts,
                relation_themes=None)
            json_results = output_handler.result_to_json()  # json字典列表
            npy_results = output_handler.result_to_npy()
            npy_results = dimension_to_3d(npy_results)
            res = []
            print(len(json_results), len(npy_results))
            for jr, nr in zip(json_results, npy_results):
                res.append({"json": jr, "npy": nr})
            send_ndarray(sink, result['client_id'], res)

    def input_fn_builder(self, socks, tf):
        import sys
        sys.path.append('..')
        from bert_cause.base.bert.tokenization import FullTokenizer

        def generator():
            tokenizer = FullTokenizer(vocab_file=self.vocab_file)

            poller = zmq.Poller()
            for sock in socks:
                poller.register(sock, zmq.POLLIN)

            logger.info('ready and listening!')
            while not self.exit_flag.is_set():
                events = dict(poller.poll())
                for sock_idx, sock in enumerate(socks):
                    if sock in events:
                        # 接收来自客户端的消息
                        client_id, raw_msg = sock.recv_multipart()
                        msg = jsonapi.loads(raw_msg)  # msg 为 json 数组
                        assert type(msg) == list
                        assert type(msg[0]) == dict
                        # for idx, text in enumerate(msg):
                        #     logger.info('[ENCODER REQUEST] WorkerID: %s, Client: %s, Received Text: %s'
                        #                 % (self.worker_id, client_id, text))
                        # 接收文本并转化为 examples
                        from bert_theme.base.input.input_handler import InputHandler
                        from bert_theme.base.model.model_feature import examples2features
                        input_handler = InputHandler()
                        self.examples = input_handler.get_pred_examples(msg)
                        # 将 examples 转化为 features
                        model_features = list(examples2features(
                            examples=self.examples,
                            max_seq_length=self.max_seq_len,
                            tokenizer=tokenizer))
                        yield {
                            'client_id': client_id,
                            'input_ids': [f.input_ids for f in model_features],
                            'input_mask': [f.input_mask for f in model_features],
                        }

        def input_fn():
            return (tf.data.Dataset.from_generator(
                generator,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'client_id': tf.string},
                output_shapes={
                    'client_id': (),
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len)
                }).prefetch(self.prefetch_size))

        return input_fn


def import_tf(device_id=-1, verbose=False, use_fp16=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf
