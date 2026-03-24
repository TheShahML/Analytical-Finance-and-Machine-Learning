"""Minimal logging configuration shared across scripts and notebooks."""

from __future__ import annotations

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a configured logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
