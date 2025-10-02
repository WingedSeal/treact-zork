import logging
import datetime
from pathlib import Path

LOG_DIRECTORY = Path("./logs")
LOG_DIRECTORY.mkdir(exist_ok=True)


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                f"{LOG_DIRECTORY}/mcp-server_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log",
                mode="a",
                encoding="utf-8",
            )
        ],
    )

    logger = logging.getLogger(__name__)
    return logger
