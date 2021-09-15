# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from server.http_server import HttpProxy
from server.args_helper import get_server_args


def start_server():
    args = get_server_args()
    proc_proxy = HttpProxy(args)
    proc_proxy.start()
    proc_proxy.join()


if __name__ == '__main__':
    start_server()
