from datetime import datetime
import logging
import time
import yaml
from pyprojroot import here


#! Load Parameters
with open(here("configs/app_config.yml")) as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)

save_logs = app_config["dev_comments"]["save_logs"]

CURRENT_DATE_Y_M_D_T_H_M_S = datetime.now().strftime("%Y-%m-%dT%H%M%S")
LOGGING_ENV_EQUIVALENCE = {
    "INFO": logging.INFO,
    "WARN": logging.WARN,
    "ERROR": logging.ERROR,
    "DEBUG": logging.DEBUG,
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "NOTSET": logging.NOTSET,
}


def _get_logging_level(level: str) -> logging:
    """
    Get the logging level from the environment variable
    """
    logging_level = LOGGING_ENV_EQUIVALENCE.get(level)
    if logging_level is None:
        raise ValueError("Logging level {} is not supported".format(level))

    return logging_level


def create_logger() -> logging.Logger:
    """
    Create a logger with the given logging level
    """
    log_level_stream = _get_logging_level(
        "DEBUG"
    )  # Options: "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"
    log_level_file = _get_logging_level(
        "INFO"
    )  # Options: "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"

    log_format = "[%(asctime)s] [%(levelname)s] [%(lineno)s] [%(module)s] %(message)s"
    log_timestamp_format = "%Y-%m-%dT%H:%M:%S%z"

    formatter = logging.Formatter(log_format, log_timestamp_format)
    formatter.converter = time.gmtime

    # Create stream handler with its specific log level
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level_stream)

    # Create logger, set its level to the lowest level to ensure all messages are processed
    default_logger = logging.getLogger(__name__)
    default_logger.setLevel(min(log_level_stream, log_level_file))
    default_logger.addHandler(stream_handler)

    # Check if logs should be saved to file
    if save_logs:
        # Create file handler with its specific log level
        file_handler = logging.FileHandler(
            f"logs/signal_track_maintenance_{CURRENT_DATE_Y_M_D_T_H_M_S}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level_file)
        default_logger.addHandler(file_handler)

    return default_logger


logger = create_logger()
