# -*- coding: utf-8 -*-

import argparse
import os

__version__ = '0.1.0'


def get_args():
    from bert_theme.base.main import __version__
    parser = argparse.ArgumentParser()
    root_path = 'C:\\Users\\10740\\IdeaProjects\\pyProjects\\BERT-CRF-CRE\\bert_theme'
    config_path = os.path.join(root_path, "config")

    group1 = parser.add_argument_group(
        'File Paths', 'config the path, checkpoint and filename of a pre-trained/fine-tuned BERT model')
    group2 = parser.add_argument_group('Model Config', 'config the model params')

    # ---------------------------------------------------------------------- #
    group1.add_argument('-init_checkpoint', type=str, default=os.path.join(config_path, 'model.ckpt-1320'))
    group2.add_argument('-do_train', type=str2bool, default=True)
    group2.add_argument('-batch_size', type=int, default=6)
    group2.add_argument('-num_train_epochs', type=float, default=10)
    group2.add_argument('-save_checkpoints_steps', type=int, default=100)
    group2.add_argument('-save_summary_steps', type=int, default=100)
    # ---------------------------------------------------------------------- #

    group1.add_argument('-data_dir', type=str, default=os.path.join(root_path, 'data'))
    group1.add_argument('-bert_config_file', type=str, default=os.path.join(config_path, 'bert_config.json'))
    group1.add_argument('-output_dir', type=str, default=os.path.join(root_path, 'result'))
    group1.add_argument('-vocab_file', type=str, default=os.path.join(config_path, 'vocab.txt'))
    group2.add_argument('-max_seq_length', type=int, default=128)
    group2.add_argument('-do_eval', type=str2bool, default=True)
    group2.add_argument('-do_predict', type=str2bool, default=True)
    group2.add_argument('-learning_rate', type=float, default=1e-5)
    group2.add_argument('-dropout_rate', type=float, default=0.5)
    group2.add_argument('-clip', type=float, default=0.5, help='Gradient clip')
    group2.add_argument('-warmup_proportion', type=float, default=0.1,
                        help='Proportion of training to perform linear learning rate warmup for '
                             'E.g., 0.1 = 10% of training.')
    group2.add_argument('-lstm_size', type=int, default=128,
                        help='size of lstm units.')
    group2.add_argument('-num_layers', type=int, default=1,
                        help='number of rnn layers, default is 1.')
    group2.add_argument('-cell', type=str, default='lstm',
                        help='which rnn cell used.')
    group2.add_argument('-filter_adam_var', type=bool, default=False,
                        help='after training do filter Adam params from model and save no Adam params model in file.')
    group2.add_argument('-do_lower_case', type=bool, default=True,
                        help='Whether to lower case the input text.')
    group2.add_argument('-clean', type=bool, default=True)
    group2.add_argument('-device_map', type=str, default='0',
                        help='witch device using to train')

    # add labels
    group2.add_argument('-label_list', type=str, default=None,
                        help='User define labels， can be a file with one label one line or a string using \',\' split')

    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    parser.add_argument('-cre', type=str, default='cre', help='which model to train')
    parser.add_argument('-version', action='version', version='%(prog)s ' + __version__)
    return parser.parse_args()


def get_ser_args():
    root_path = '.'

    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('File Paths',
                                       'config the path, checkpoint and filename of a pertained/fine-tuned BERT model')
    group1.add_argument('-root_path', type=str, default=root_path,
                        help='absolute work dir. Example: /root/bert/bert_cause')
    group1.add_argument('-init_checkpoint', type=str, default='model.ckpt-4788')
    group1.add_argument('-output_dir', type=str, default='result')
    group1.add_argument('-vocab_file', type=str, default='vocab.txt')
    group1.add_argument('-model_dir', type=str, default='',
                        help='directory of a pertrained BERT model')
    group1.add_argument('-tuned_model_dir', type=str,
                        default='',
                        help='directory of a fine-tuned BERT model')
    group1.add_argument('-ckpt_name', type=str, default='model.ckpt-1841',
                        help='filename of the checkpoint file. By default it is "bert_model.ckpt", but \
                             for a fine-tuned model the name could be different.')
    group1.add_argument('-config_name', type=str, default='bert_config.json',
                        help='filename of the JSON config file for BERT model.')
    group1.add_argument('-bert_config_file', type=str, default='bert_config.json')

    group2 = parser.add_argument_group('BERT Parameters',
                                       'config how BERT model and pooling works')
    group2.add_argument('-max_seq_len', type=int, default=128,
                        help='maximum length of a sequence')
    group2.add_argument('-pooling_layer', type=int, nargs='+', default=[-2],
                        help='the encoder layer(s) that receives pooling. \
                        Give a list in order to concatenate several layers into one')
    group2.add_argument('-mask_cls_sep', action='store_true', default=False,
                        help='masking the embedding on [CLS] and [SEP] with zero. \
                        When pooling_strategy is in {CLS_TOKEN, FIRST_TOKEN, SEP_TOKEN, LAST_TOKEN} \
                        then the embedding is preserved, otherwise the embedding is masked to zero before pooling')
    group2.add_argument('-lstm_size', type=int, default=128,
                        help='size of lstm units.')

    group3 = parser.add_argument_group('Serving Configs',
                                       'config how server utilizes GPU/CPU resources')
    group3.add_argument('-port', '-port_in', '-port_data', type=int, default=5557,
                        help='server port for receiving data from client')
    group3.add_argument('-port_out', '-port_result', type=int, default=5558,
                        help='server port for sending result to client')
    group3.add_argument('-host', type=str, default='localhost',
                        help='server host for receiving HTTP requests')
    group3.add_argument('-timeout', type=int, default=-1,
                        help='maximum time for the server to response, set -1 to be unlimited.')
    group3.add_argument('-http_port', type=int, default=8555,
                        help='server port for receiving HTTP requests')
    group3.add_argument('-http_max_connect', type=int, default=10,
                        help='maximum number of concurrent HTTP connections')
    group3.add_argument('-cors', type=str, default='*',
                        help='setting "Access-Control-Allow-Origin" for HTTP requests')
    group3.add_argument('-num_worker', type=int, default=1,
                        help='number of server instances')
    group3.add_argument('-max_batch_size', type=int, default=1024,
                        help='maximum number of sequences handled by each worker')
    group3.add_argument('-priority_batch_size', type=int, default=16,
                        help='batch smaller than this size will be labeled as high priority,'
                             'and jumps forward in the job queue')
    group3.add_argument('-cpu', type=str2bool, default=True,
                        help='running on CPU (default on GPU)')
    group3.add_argument('-xla', type=str2bool, default=False,
                        help='enable XLA compiler (experimental)')
    group3.add_argument('-fp16', type=str2bool, default=False,
                        help='use float16 precision (experimental)')
    group3.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determine the fraction of the overall amount of memory \
                        that each visible GPU should be allocated per worker. \
                        Should be in range [0.0, 1.0]')
    group3.add_argument('-device_map', type=int, nargs='+', default=[],
                        help='specify the list of GPU device ids that will be used (id starts from 0). \
                        If num_worker > len(device_map), then device will be reused; \
                        if num_worker < len(device_map), then device_map[:num_worker] will be used')
    group3.add_argument('-prefetch_size', type=int, default=10,
                        help='the number of batches to prefetch on each worker. When running on a CPU-only machine, \
                        this is set to 0 for comparability')

    parser.add_argument('-verbose', type=str2bool, default=False,
                        help='turn on tensorflow logging for debug')
    # parser.add_argument('-mode', type=str, default='NER')
    parser.add_argument('-version', action='version', version='%(prog)s ' + __version__)
    return parser.parse_args()


def get_server_args():
    args = get_ser_args()
    args.root_path = os.path.join(args.root_path, 'bert_theme')
    config_path = os.path.join(args.root_path, 'config')
    args.output_dir = os.path.join(args.root_path, args.output_dir)
    args.init_checkpoint = os.path.join(config_path, args.init_checkpoint)
    args.vocab_file = os.path.join(config_path, args.vocab_file)
    args.model_dir = config_path
    args.tuned_model_dir = config_path
    args.bert_config_file = os.path.join(config_path, args.bert_config_file)
    return args


def print_args(args):
    import sys
    param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
    print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))


def str2bool(s):
    return True if str(s).lower() == 'true' else False
