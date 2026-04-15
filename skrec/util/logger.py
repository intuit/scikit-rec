import logging
from enum import Enum
from logging import Logger


class ExtraLoggingFields(str, Enum):
    LOG_TAG = "log_tag"


class ExtraFormatter(logging.Formatter):
    def __init__(self):
        format = "%(asctime)s - %(name)s - %(levelname)s %(EXTRA_PLACEHOLDER)s%(message)s"
        super().__init__(fmt=format)

    def format(self, record):
        extra_msg = ""
        for custom_var in ExtraLoggingFields:
            if hasattr(record, custom_var.value):
                extra_msg += f"- {str(getattr(record, custom_var.value))} "
                delattr(record, custom_var.value)
        setattr(record, "EXTRA_PLACEHOLDER", extra_msg)
        return logging.Formatter.format(self, record)


class LogType(str, Enum):
    EVALUATION = "EVALUATION"
    TRAIN = "TRAIN"
    PREDICT = "PREDICT"
    METRIC = "METRIC"


def get_logger(name: str) -> Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(ExtraFormatter())
        logger.addHandler(handler)

    return logger
