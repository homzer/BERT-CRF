# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def run_model(printed=True):
    import os
    from bert_cause.base.helper.args_helper import get_train_args, print_args
    from bert_cause.base.main import run_bert

    args = get_train_args()
    if printed:
        print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    run_bert(args=args)


if __name__ == '__main__':
    run_model()
