# -*- coding: UTF-8 -*-
import pickle
import uuid
from collections import namedtuple

import zmq
from zmq.utils import jsonapi

from bert_cause.base.helper.log_helper import get_logger

logger = get_logger()

Response = namedtuple('Response', ['id', 'content'])
# in the future client version must match with server version
__version__ = '0.1.0'


def timeout(func):
    def arg_wrapper(self, *args, **kwargs):
        if 'blocking' in kwargs and not kwargs['blocking']:
            # override client timeout setting if `func` is called in non-blocking way
            self.receiver.setsockopt(zmq.RCVTIMEO, -1)
        else:
            self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
        try:
            return func(self, *args, **kwargs)
        except zmq.error.Again as _e:
            print('No response from the server (With "timeout"=%d ms).' % self.timeout)
            exit(0)
        finally:
            self.receiver.setsockopt(zmq.RCVTIMEO, -1)

    return arg_wrapper


class TcpClient(object):
    def __init__(self, ip='localhost', port=5557, port_out=5558,
                 identity=None,
                 exceed_time=-1):

        self.context = zmq.Context()

        self.sender = self.context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.sender.connect('tcp://%s:%d' % (ip, port))

        self.identity = identity or str(uuid.uuid4())[:8].encode('ascii')

        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.identity)
        self.receiver.connect('tcp://%s:%d' % (ip, port_out))

        self.request_id = 0
        self.timeout = exceed_time
        self.pending_request = set()

        self.port = port
        self.port_out = port_out
        self.ip = ip
        self.length_limit = 0

        logger.info("A new Tcp Client is created!")
        # self._send(b'SHOW_CONFIG')
        # logger.info(jsonapi.loads(self._recv().content[1]))

    def close(self):
        """
            Gently close all connections of the client. If you are using BertClient as context manager,
            then this is not necessary.
        """
        self.sender.close()
        self.receiver.close()
        self.context.term()

    def _send(self, msg, msg_len=0):
        self.sender.send_multipart([self.identity, msg, b'%d' % self.request_id, b'%d' % msg_len])
        self.pending_request.add(self.request_id)
        self.request_id += 1

    def _recv(self):
        response = self.receiver.recv_multipart()
        request_id = int(response[-1])
        self.pending_request.remove(request_id)
        return Response(request_id, response)

    def _recv_ndarray(self):
        request_id, response = self._recv()
        arr_val = pickle.loads(response[1])
        return Response(request_id, arr_val)

    @timeout
    def request(self, texts, blocking=True):
        self._send(jsonapi.dumps(texts), len(texts))
        rst = self._recv_ndarray().content if blocking else None
        return rst

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
