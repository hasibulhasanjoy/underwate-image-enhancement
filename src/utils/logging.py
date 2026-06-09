"""
src/utils/logging.py
--------------------
Centralised logging setup for the whole project.

Responsibilities:
  - Provide a factory function that returns a logger writing to
    both the console (coloured) and a rotating file.
  - One call per module: ``logger = get_logger(__name__)``
  - Console output is human-friendly; file output is JSON-like for
    easy parsing by monitoring tools.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# ANSI colour codes for console handler
# ---------------------------------------------------------------------------
_COLOURS = {
    logging.DEBUG:    "\033[36m",   # cyan
    logging.INFO:     "\033[32m",   # green
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = _COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    *,
    log_dir: Optional[str] = "logs",
    log_file: Optional[str] = "pipeline.log",
    level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Return a project logger with console + rotating-file handlers.

    The function is idempotent: calling it twice with the same *name*
    returns the same logger without adding duplicate handlers.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.
    log_dir:
        Directory for the log file.  Created if it does not exist.
        Pass ``None`` to disable file logging.
    log_file:
        File name inside *log_dir*.
    level:
        Minimum severity level forwarded to all handlers.
    max_bytes:
        Maximum size of the log file before rotation.
    backup_count:
        Number of rotated files to keep.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers more than once (e.g. in Jupyter / reload)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ── Console handler ────────────────────────────────────────────────────
    console_fmt = _ColourFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # ── Rotating file handler ──────────────────────────────────────────────
    if log_dir is not None and log_file is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_fmt = logging.Formatter(
            fmt='{"time": "%(asctime)s", "level": "%(levelname)s", '
                '"module": "%(name)s", "message": "%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            log_path / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_fmt)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger