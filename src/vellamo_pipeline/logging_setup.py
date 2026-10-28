"""Structured JSON logging to stdout and a rotating file (swap in your Day 35 module if you prefer)."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_dir: Path, level: int = logging.INFO, filename: str = "pipeline.log") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormatter()
    handlers = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(log_dir / filename, maxBytes=1_000_000, backupCount=3, encoding="utf-8"),
    ]
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    # httpx logs every full URL at INFO; our own api_attempt records already cover this.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    