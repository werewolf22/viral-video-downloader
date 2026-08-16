"""Shared helpers: logging, workspace state, filesystem utilities."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-12s  %(message)s"
_configured = False


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once, idempotently."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_LOG_FORMAT,
        datefmt="%H:%M:%S",
    )
    # yt-dlp and urllib3 are chatty at DEBUG.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def slugify(value: str, max_length: int = 60) -> str:
    """Filesystem-safe ASCII slug. Titles come from the internet, so be strict."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    value = re.sub(r"[\s_-]+", "-", value)
    return value[:max_length].strip("-") or "untitled"


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    # Write via a temp file so an interrupted run never leaves a truncated state file.
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def human_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def srt_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` — the only form SRT parsers accept."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class StageError(RuntimeError):
    """Raised when a pipeline stage cannot produce its output."""
