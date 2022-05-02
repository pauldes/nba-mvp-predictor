import logging
import os
import random

import numpy

from nba_mvp_predictor import utils

SEED = 666


def get_conf():
    return utils.get_dict_from_yaml("nba_mvp_predictor/conf.yaml")

def seed_packages(seed: int = SEED):
    """Set seed for all packages using pseudorandom generation.

    Args:
        seed (int, optional): Seed number. Defaults to SEED.

    Returns:
        int: Seed number used.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    return seed

def get_logger():
    log_level = os.environ.get("LOG_LEVEL", "DEBUG")
    if log_level == "DEBUG":
        level = logging.DEBUG
    elif log_level == "INFO":
        level = logging.INFO
    elif log_level == "WARNING":
        level = logging.WARNING
    elif log_level == "ERROR":
        level = logging.ERROR
    elif log_level == "CRITICAL":
        level = logging.CRITICAL
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger


logger = get_logger()
conf = get_conf()
