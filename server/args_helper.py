# -*- coding: UTF-8 -*-
import argparse


def get_server_args():

    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('Serving Configs')
    group1.add_argument('-theme_port', type=int, default=5557)
    group1.add_argument('-theme_port_out', type=int, default=5558)
    group1.add_argument('-cause_port', type=int, default=5555)
    group1.add_argument('-cause_port_out', type=int, default=5556)
    group1.add_argument('-host', type=str, default='localhost')
    group1.add_argument('-timeout', type=int, default=-1)
    group1.add_argument('-http_port', type=int, default=8555)
    group1.add_argument('-cors', type=str, default='*')

    return parser.parse_args()
