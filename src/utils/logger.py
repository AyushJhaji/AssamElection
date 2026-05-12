"""src/utils/logger.py — Structured logger for the pipeline."""

import logging
import sys
from pathlib import Path


def setup(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logger. Call once from main.py."""
    fmt = "%(asctime)s  %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)
