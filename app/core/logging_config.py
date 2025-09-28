# app/core/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import os

LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def configure_logging():
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # rotating file
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=3)
    fh.setFormatter(formatter)
    root.addHandler(fh)
