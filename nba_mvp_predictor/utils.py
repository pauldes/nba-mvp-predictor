import random
import time

import box
import yaml


def get_dict_from_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        conf_dict = yaml.safe_load(f)
    conf_dict = box.Box(conf_dict, default_box=True, default_box_attr=None)
    return conf_dict


def wait_random_time(min, max) -> None:
    time.sleep(random.randint(min, max))
