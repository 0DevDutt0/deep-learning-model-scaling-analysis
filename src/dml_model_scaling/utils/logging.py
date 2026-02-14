"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path to write logs to.
        format_string: Optional custom format string for log messages.

    Returns:
        Configured logger instance.

    Examples:
        >>> logger = setup_logging(level="DEBUG")
        >>> logger.info("Application started")
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("dml_model_scaling")
    logger.setLevel(getattr(logging, level.upper()))

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Name of the logger (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"dml_model_scaling.{name}")
