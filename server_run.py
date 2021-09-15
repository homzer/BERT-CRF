# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from bert_cause.base.server.tcp_server import run_tcp_server
from bert_cause.base.helper.args_helper import get_server_args, print_args


def start_server(printed=True):
    args = get_server_args()
    if printed:
        print_args(args)
    run_tcp_server(args)


if __name__ == '__main__':
    start_server()
