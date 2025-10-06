import datetime
import logging
import os
import sys
from pathlib import Path
from typing import cast


LOG_DIRECTORY = Path("./logs")
LOG_DIRECTORY.mkdir(exist_ok=True)

FORMATTER = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
CONSOLE_FORMATTER = logging.Formatter("%(name)s | %(levelname)s | %(message)s")


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
console_log_level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
file_log_level = os.getenv("FILE_LOG_LEVEL", "DEBUG").upper().strip()
log_level = LOG_LEVELS.get(console_log_level, logging.INFO)

VALID_FILE_LOG_LEVELS = ("NONE", "DEBUG", "INFO")


class CustomLogger(logging.Logger):
    def should_log_to_stdio(self, log_level: int) -> bool:
        """Check if any StreamHandler writing to stdout/stderr would log the log_level messages"""
        if not self.isEnabledFor(log_level):
            return False

        for handler in self.handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream in (sys.stdout, sys.stderr):
                    if handler.level <= log_level:
                        return True
        return False


def get_logger(name: str) -> CustomLogger:
    """
    Usage:
    ```python
    logger = get_logger(__name__)
    ```
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(CONSOLE_FORMATTER)
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
    logger.__class__ = CustomLogger
    return cast(CustomLogger, logger)


logger = get_logger(__name__)


class InvalidEnvironmentException(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


if console_log_level not in LOG_LEVELS:
    error = f"Invalid LOG_LEVEL: {console_log_level}. LOG_LEVEL options: {', '.join(LOG_LEVELS)}"
    logger.critical(error)
    raise InvalidEnvironmentException(error)
if file_log_level not in VALID_FILE_LOG_LEVELS:
    error = f"Invalid FILE_LOG_LEVEL: {file_log_level}. FILE_LOG_LEVEL options: {', '.join(VALID_FILE_LOG_LEVELS)}"
    logger.critical(error)
    raise InvalidEnvironmentException(error)
