import random

import box
import yaml


def get_dict_from_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        conf_dict = yaml.safe_load(f)
    conf_dict = box.Box(conf_dict, default_box=True, default_box_attr=None)
    return conf_dict


def sample_uniform_seconds(low: float = 0, high: float = 1) -> float:
    """Sample a uniformly random duration in ``[min(low, high), max(low, high)]`` (seconds)."""
    a, b = min(low, high), max(low, high)
    return random.uniform(a, b)
