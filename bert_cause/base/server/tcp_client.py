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
    def __init__(self, ip='localhost', port=5555, port_out=5556,
                 output_fmt='ndarray', show_server_config=False,
                 identity=None, check_length=True,
                 exceed_time=-1):
        """ A client object connected to a BertServer

        Create a BertClient that connects to a BertServer.
        Note, server must be ready at the moment you are calling this function.
        If you are not sure whether the server is ready, then please set `check_version=False` and `check_length=False`

        Example usage:
            client = BertClient(show_server_config=True, check_length=False, exceed_time=6000)
            str1 = '公司的运营和收入受到了疫情影响'
            rst = client.request([str1])
            print('rst:', rst)

        :type exceed_time: int
        :type check_length: bool
        :type identity: str
        :type show_server_config: bool
        :type output_fmt: str
        :type port_out: int
        :type port: int
        :type ip: str
        :param ip: the ip address of the server
        :param port: port for pushing data from client to server, must be consistent with the server side config
        :param port_out: port for publishing results from server to client, must be consistent with the server side config
        :param output_fmt: the output format of the sentence encodes, either in numpy array or python List[List[float]] (ndarray/list)
        :param show_server_config: whether to show server configs when first connected
        :param identity: the UUID of this client
        :param check_length: check if server `max_seq_len` is less than the sentence length before sent
        :param exceed_time: set the timeout (milliseconds)
        for receive operation on the client, -1 means no timeout and wait until result returns
        """

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

        if output_fmt == 'ndarray':
            self.formatter = lambda x: x
        elif output_fmt == 'list':
            self.formatter = lambda x: x.tolist()
        else:
            raise AttributeError('"output_fmt" must be "ndarray" or "list"')

        self.output_fmt = output_fmt
        self.port = port
        self.port_out = port_out
        self.ip = ip
        self.length_limit = 0

        logger.info("A new Tcp Client is created!")
        s_status = self.server_status
        if show_server_config:
            self._print_dict(s_status, 'server config:')

        if check_length:
            self.length_limit = int(s_status['max_seq_len'])

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

    @property
    def status(self):
        """
            Get the status of this BertClient instance
        :rtype: dict[str, str]
        :return: a dictionary contains the status of this BertClient instance

        """
        return {
            'identity': self.identity,
            'num_request': self.request_id,
            'num_pending_request': len(self.pending_request),
            'pending_request': self.pending_request,
            'output_fmt': self.output_fmt,
            'port': self.port,
            'port_out': self.port_out,
            'server_ip': self.ip,
            'client_version': __version__,
            'timeout': self.timeout
        }

    @property
    @timeout
    def server_status(self):
        """
            Get the current status of the server connected to this client
        :return: a dictionary contains the current status of the server connected to this client
        :rtype: dict[str, str]
        """
        self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
        self._send(b'SHOW_CONFIG')
        return jsonapi.loads(self._recv().content[1])

    @timeout
    def request(self, texts, blocking=True):
        self._send(jsonapi.dumps(texts), len(texts))
        rst = self._recv_ndarray().content if blocking else None
        return rst

    @staticmethod
    def _check_length(texts, len_limit, tokenized):
        if tokenized:
            # texts is already tokenized as list of str
            return all(len(t) <= len_limit for t in texts)
        else:
            # do a simple whitespace tokenizer
            return all(len(t.split()) <= len_limit for t in texts)

    @staticmethod
    def _print_dict(x, title=None):
        if title:
            print(title)
        for k, v in x.items():
            print('%30s\t=\t%-30s' % (k, v))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
