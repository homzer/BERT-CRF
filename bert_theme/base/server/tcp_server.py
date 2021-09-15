# -*- coding: UTF-8 -*-
import random
import sys
import threading
from datetime import datetime
from multiprocessing.pool import Pool

import zmq
import zmq.decorators as zmqd
from zmq.utils import jsonapi

from bert_theme.base.helper.label_helper import init_label
from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.main import __version__
from bert_cause.base.server.bert_sink import BertSink
from bert_cause.base.server.server_helper import ServerCommand, auto_bind

from bert_theme.base.server.bert_worker import BertWorker

from bert_cause.base.server.server_status import ServerStatus
from bert_cause.base.server.zmq_decor import multi_socket

logger = get_logger()


def check_tf_version():
    import tensorflow as tf
    tf_ver = tf.__version__.split('.')
    assert int(tf_ver[0]) >= 1 and int(tf_ver[1]) >= 10, 'Tensorflow >=1.10 is required!'
    return tf_ver


_tf_ver_ = check_tf_version()


class TcpServer(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self)
        self.logger = get_logger()

        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.num_concurrent_socket = max(8, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port
        self.args = args
        self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
        self.status_static = {
            'tensorflow_version': _tf_ver_,
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }
        self.processes = []

        self.num_labels = 0
        self.id2theme = dict()
        self.graph_path = None
        self.load_theme_model()

    def load_theme_model(self):
        self.logger.info('Loading model, could take a while...')
        with Pool(processes=1) as pool:
            # optimize the graph, must be done in another process
            from .graph import optimize_model
            num_labels, id2theme = init_label(self.args.output_dir)
            self.num_labels = num_labels
            self.id2theme = id2theme
            self.graph_path = pool.apply(optimize_model, (self.args, self.num_labels))
        if self.graph_path:
            self.logger.info('optimized graph is stored at: %s' % self.graph_path)
        else:
            raise FileNotFoundError('graph optimization fails and returns empty result')

    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        for p in self.processes:
            p.close()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'', ServerCommand.terminate, b'', b''])

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, frontend, sink, *backend_socks):

        def push_new_job(_job_id, _json_msg, _msg_len):
            # backend_socks[0] is always at the highest priority
            _sock = backend_socks[0] if _msg_len <= self.args.priority_batch_size else rand_backend_socket
            _sock.send_multipart([_job_id, _json_msg])

        # bind all sockets
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d ventilator-worker sockets, %s'
                         % (len(addr_backend_list), ', '.join(addr_backend_list)))

        # start the sink process
        # sink是用来接收上层BertWork的产出，然后发送给client
        self.logger.info('start the sink')
        proc_sink = BertSink(self.args, addr_front2sink)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        # 这里启动多个进程，加载主模型
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):
            process = BertWorker(idx, self.args, addr_backend_list, addr_sink, device_id,
                                 self.graph_path, self.id2theme)
            self.processes.append(process)
            process.start()

        # start the http-service process
        # if self.args.http_port:
        #     from bert_theme.base.server.http_server import HttpProxy
        #     self.logger.info('starting http proxy...')
        #     proc_proxy = HttpProxy(self.args)
        #     self.processes.append(proc_proxy)
        #     proc_proxy.start()

        rand_backend_socket = None
        server_status = ServerStatus()
        while True:
            request = frontend.recv_multipart()
            try:
                client, msg, req_id, msg_len = request
            except ValueError:
                self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
            else:
                server_status.update(request)
                if msg == ServerCommand.terminate:
                    break
                elif msg == ServerCommand.show_config:
                    self.logger.info('[CONFIG REQUEST] RequestID: %d  Client: %s' % (int(req_id), client))
                    sink.send_multipart([client,
                                         msg,
                                         jsonapi.dumps({**self.status_args, **self.status_static}),
                                         req_id])
                else:
                    self.logger.info('[ENCODER REQUEST] RequestID: %d  Client: %s  ContentSize: %d' %
                                     (int(req_id), client, int(msg_len)))
                    sink.send_multipart([client, ServerCommand.new_job, msg_len, req_id])  # 发送给 bert sink
                    rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])
                    job_id = client + b'#' + req_id
                    if int(msg_len) > self.max_batch_size:
                        seqs = jsonapi.loads(msg)
                        job_gen = ((job_id + b'@%d' % i, seqs[i:(i + self.max_batch_size)]) for i in
                                   range(0, int(msg_len), self.max_batch_size))
                        for partial_job_id, job in job_gen:
                            push_new_job(partial_job_id, jsonapi.dumps(job), len(job))
                    else:
                        push_new_job(job_id, msg, int(msg_len))

        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.args.cpu:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker))
                num_avail_gpu = len(avail_gpu)

                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' %
                                        (num_avail_gpu, num_all_gpu, self.num_worker))
                    if not self.args.device_map:
                        self.logger.warning('multiple workers will be allocated to one GPU, '
                                            'may not scale well and may raise out-of-memory')
                    else:
                        self.logger.warning('workers will be allocated based on "-device_map=%s", '
                                            'may not scale well and may raise out-of-memory' % self.args.device_map)
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: %s' % ', '.join(
            'worker %d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map

    def append_process(self, process):
        self.processes.append(process)


def run_tcp_server(args):
    server = TcpServer(args)
    server.start()
    server.join()
