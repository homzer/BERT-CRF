# -*- coding: UTF-8 -*-
import os
import pickle
import uuid

import numpy
import zmq

from bert_cause.base.helper.log_helper import get_logger

logger = get_logger()


class ServerCommand:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCommand).items())


def send_ndarray(src, dest, x, req_id=b'', flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    if type(x) is not list and type(x) is not numpy.ndarray:
        raise TypeError("x must be list or numpy.ndarray, got %s" % type(x))
    # logger.info("client: %s  sink: %s  msg: %s" % (dest, src, x))
    return src.send_multipart([dest, pickle.dumps(x), req_id], flags, copy=copy, track=track)


def auto_bind(socket):
    """
    自动进行端口绑定
    :param socket:
    :return:
    """
    if os.name == 'nt':  # for Windows
        socket.bind_to_random_port('tcp://127.0.0.1')
    else:
        # Get the location for tmp file for sockets
        try:
            tmp_dir = os.environ['ZEROMQ_SOCK_TMP_DIR']
            if not os.path.exists(tmp_dir):
                raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
            # 随机产生一个
            tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
        except KeyError:
            tmp_dir = '*'

        socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')
