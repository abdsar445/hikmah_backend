"""
app/core/logger.py
------------------
Centralised logging configuration.
All modules import `logger` from here so the format is consistent.
"""

import logging
import sys


def _build_logger(name: str = "himak") -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:          # avoid adding duplicate handlers on reload
        return log

    log.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    log.addHandler(handler)
    return log


logger = _build_logger()