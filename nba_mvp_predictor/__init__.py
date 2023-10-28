import logging
import os
import random

import numpy

from nba_mvp_predictor import utils

SEED = 0


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


def build_logger() -> logging.Logger:
    """Configure logger for the project

    Returns:
        logging.Logger: Logger to use for the project
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
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
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-12s %(message)s"
    )
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = build_logger()
conf = get_conf()
seed = seed_packages()
