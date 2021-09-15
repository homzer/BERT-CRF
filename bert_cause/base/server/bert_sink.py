# -*- coding: UTF-8 -*-
import multiprocessing
import pickle
import time
from collections import defaultdict
from multiprocessing import Process

import numpy as np
import zmq
import zmq.decorators as zmqd

from bert_cause.base.helper.log_helper import get_logger
from bert_cause.base.server.server_helper import ServerCommand, send_ndarray, auto_bind

logger = get_logger()


class BertSink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.front_sink_addr = front_sink_addr
        self.verbose = args.verbose
        self.args = args

    def close(self):
        logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        pending_checksum = defaultdict(int)
        pending_result = defaultdict(list)
        job_checksum = defaultdict(int)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))
        logger.info('ready')
        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                job_id = msg[0]
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = job_info[1] if len(job_info) == 2 else 0

                # parsing the ndarray
                arr_val = pickle.loads(msg[1])
                logger.info("arr_val: %s" % arr_val)
                pending_result[job_id].append((arr_val, partial_id))

                pending_checksum[job_id] += len(arr_val)
                # pending_checksum[job_id] = job_checksum[job_id]

                # logger.info("pending_result: %s" % pending_result)
                # logger.info("pending_checksum: %s" % pending_checksum)
                # logger.info("job_checksum: %s" % job_checksum)

                # check if there are finished jobs, send it back to workers
                # 这里十分关键，关系到 tcp client 是否能接收到处理完成的结果
                finished = []
                for (k, v) in pending_result.items():
                    if pending_checksum[k] >= job_checksum[k] or pending_checksum[k] > 200:
                        finished.append((k, v))
                    else:
                        logger.info("Job %s dose not finish yet. all: %d  done: %d"
                                    % (k, job_checksum[k], pending_checksum[k]))
                # logger.info("finished: %s" % finished)
                for job_info, tmp in finished:
                    # logger.info('[ENCODER REQUEST] Send back to client. JobID: %s' % job_info)
                    # re-sort to the original order
                    tmp = [x[0] for x in sorted(tmp, key=lambda x: int(x[1]))]
                    # logger.info("tmp: %s  shape: %s" % (tmp, np.shape(tmp)))
                    client_addr, req_id = job_info.split(b'#')
                    tmp = np.concatenate(tmp, axis=0)
                    # logger.info("concat tmp: %s  shape: %s" % (tmp, np.shape(tmp)))
                    send_ndarray(sender, client_addr, tmp, req_id)
                    pending_result.pop(job_info)
                    pending_checksum.pop(job_info)
                    job_checksum.pop(job_info)

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == ServerCommand.new_job:
                    job_info = client_addr + b'#' + req_id
                    job_checksum[job_info] = int(msg_info)
                    logger.info('[ENCODER REQUEST] Encoder Job Registered. JobID: %s' % job_info)
                elif msg_type == ServerCommand.show_config:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('[CONFIG REQUEST] Server Config Sent.')
                    sender.send_multipart([client_addr, msg_info, req_id])
