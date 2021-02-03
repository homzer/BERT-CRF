from bert_base.server.helper import set_logger
from bert_base.train.me.processor import NerProcessor

logger = set_logger('NER Load Data')


def get_labels(output_dir):
    processor = NerProcessor(output_dir)
    return processor.get_labels()


def get_test_examples(output_dir, data_dir):
    processor = NerProcessor(output_dir)
    return processor.get_test_examples(data_dir)


def load_data(args):
    processor = NerProcessor(args.output_dir)
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(len(train_examples) * 1.0 / args.batch_size * args.num_train_epochs)
    if num_train_steps < 1:
        raise AttributeError('training data is so small...')
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    eval_examples = processor.get_dev_examples(args.data_dir)

    # 打印验证集数据信息
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.batch_size)

    return train_examples, num_train_steps, num_warmup_steps, eval_examples
