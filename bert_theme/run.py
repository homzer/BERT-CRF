
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def run_model():
    import os
    from bert_theme.base.helper.args_helper import get_args
    from bert_theme.base.main import run_bert

    args = get_args()
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    run_bert(args=args)


if __name__ == '__main__':
    run_model()
