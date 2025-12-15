import logging
import sys
from pathlib import Path

def setup_logger(name: str = "agent", log_file: str = "agent.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
