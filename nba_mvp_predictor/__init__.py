import logging
import os

import yaml
import box

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
print("Log level:", log_level, level)

with open("nba_mvp_predictor/conf.yaml", "r", encoding="utf-8") as f:
    conf_dict = yaml.safe_load(f)
conf = box.Box(conf_dict)
