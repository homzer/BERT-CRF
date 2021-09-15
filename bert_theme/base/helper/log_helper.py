import logging
LOG_FORMAT = "%(asctime)s [%(filename)s:%(lineno)d]: %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"


def get_logger():
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger
