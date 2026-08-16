"""Viral compilation pipeline.

Stage modules are runnable both as a library and as scripts::

    python -m scripts.discover
    python -m scripts.fetch

The orchestrator in ``main.py`` chains them together.
"""

__all__ = [
    "config",
    "util",
    "ffmpeg_utils",
    "discover",
    "fetch",
    "highlight",
    "process",
    "subtitles",
    "editor",
    "metadata",
]
