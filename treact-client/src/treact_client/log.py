import logging
import datetime
import sys
import os
from pathlib import Path

LOG_DIRECTORY = Path("./logs")
LOG_DIRECTORY.mkdir(exist_ok=True)

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")


def get_current_time_string() -> str:
    return datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


current_time_string = get_current_time_string()

logger_file_info = LOG_DIRECTORY / f"mcp-server_{current_time_string}.log"
logger_file_debug = LOG_DIRECTORY / f"mcp-server_{current_time_string}_debug.log"

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
console_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
file_log_level = os.getenv("FILE_LOG_LEVEL", "DEBUG").upper()
log_level = LOG_LEVELS.get(console_log_level, logging.INFO)

VALID_FILE_LOG_LEVELS = ("NONE", "DEBUG", "INFO")


def get_logger(name: str) -> logging.Logger:
    """
    Usage:
    ```python
    logger = get_logger(__name__)
    ```
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(FORMATTER)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    if file_log_level in ("DEBUG", "INFO"):
        file_handler = logging.FileHandler(
            logger_file_info,
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(FORMATTER)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    if file_log_level == "DEBUG":
        file_handler_debug = logging.FileHandler(
            logger_file_debug,
            mode="a",
            encoding="utf-8",
        )
        file_handler_debug.setFormatter(FORMATTER)
        file_handler_debug.setLevel(logging.DEBUG)
        logger.addHandler(file_handler_debug)

    logger.propagate = False
    return logger


logger = get_logger(__name__)
if console_log_level not in LOG_LEVELS:
    error = f"Invalid LOG_LEVEL: {console_log_level}. LOG_LEVEL options: {', '.join(LOG_LEVELS)}"
    logger.critical(error)
    raise ValueError(error)
if file_log_level not in VALID_FILE_LOG_LEVELS:
    error = f"Invalid FILE_LOG_LEVEL: {file_log_level}. FILE_LOG_LEVEL options: {', '.join(VALID_FILE_LOG_LEVELS)}"
    logger.critical(error)
    raise ValueError(error)
