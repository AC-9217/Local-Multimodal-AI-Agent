import logging
import sys
from pathlib import Path

def setup_logger(name: str = "agent", log_file: str = "agent.log", level=logging.INFO):
    """
    Setup and return a logger with console and file handlers.
    配置并返回一个包含控制台和文件处理程序的日志记录器。

    Args:
        name (str): Name of the logger. 日志记录器的名称。
        log_file (str): Path to the log file. 日志文件的路径。
        level: Logging level. 日志级别。
    
    Returns:
        logging.Logger: Configured logger. 配置好的日志记录器。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times
    # 防止重复添加处理程序
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    # 控制台处理程序
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    # 文件处理程序
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
