# -*- coding: utf-8 -*-

import logging


def setup_logger(logging_name):

    logger = logging.getLogger(name=logging_name)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)8s %(levelname)8s: %(message)s',
        datefmt='%Y/%m/%d %p %I:%M:%S,',)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
