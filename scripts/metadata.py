"""Stage 6 - upload metadata and the attribution block.

Credits are not decoration. Every clip in the compilation belongs to someone
else, and the description generated here is what names them.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .process import Clip
from .util import get_logger, human_duration

log = get_logger("metadata")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "is",
    "it", "this", "that", "at", "by", "from", "my", "you", "your", "we", "i",
    "how", "what", "when", "official", "video", "shorts", "vs",
}

BASE_TAGS = ["compilation", "viral", "trending", "shorts", "funny", "best moments"]


@dataclass
class Metadata:
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    credits: list[str] = field(default_factory=list)


def _keywords(clips: list[Clip], limit: int = 10) -> list[str]:
    counter: Counter[str] = Counter()
    for clip in clips:
        for word in re.findall(r"[a-zA-Z']{3,}", clip.title.lower()):
            if word not in STOPWORDS:
                counter[word] += 1
    return [word for word, _ in counter.most_common(limit)]


def build_title(clips: list[Clip], cfg: Config) -> str:
    headline = cfg.branding.title_card_text.replace("\n", " ").strip()
    top = _keywords(clips, 3)
    suffix = " ".join(w.capitalize() for w in top)
    title = f"{headline} | {len(clips)} Clips" + (f" | {suffix}" if suffix else "")
    return title[:100]  # YouTube truncates titles past 100 characters


def build_metadata(clips: list[Clip], cfg: Config, duration: float = 0.0) -> Metadata:
    credits = []
    for index, clip in enumerate(clips, 1):
        channel = clip.channel or "Unknown creator"
        link = clip.channel_url or clip.source_url
        if clip.platform == "twitch":
            licence_label = "Twitch clip"
        elif clip.platform == "instagram":
            licence_label = "Instagram clip"
        elif clip.license in ("creativeCommon", "youtube-cc"):
            licence_label = "CC BY"
        else:
            licence_label = "standard licence"
        credits.append(f"{index}. {clip.title}\n   {channel} - {link}\n   Source: {clip.source_url} ({licence_label})")

    lines = [
        f"{len(clips)} of this week's most-shared moments, cut back to back.",
        "",
        f"Runtime: {human_duration(duration)}" if duration else "",
        "",
        "TIMESTAMPS",
    ]

    # Chapter markers, offset by the title card that precedes clip 1.
    position = cfg.branding.title_card_seconds if cfg.branding.title_card_text else 0.0
    for index, clip in enumerate(clips, 1):
        lines.append(f"{human_duration(position)} - #{index} {clip.title[:60]}")
        position += clip.duration

    lines += [
        "",
        "CREDITS - all clips belong to their original creators:",
        "",
        *credits,
        "",
        "If you are a creator featured here and would like your clip removed,",
        "get in touch and it will be taken down.",
    ]

    tags = BASE_TAGS + _keywords(clips, 12)
    return Metadata(
        title=build_title(clips, cfg),
        description="\n".join(line for line in lines if line is not None),
        tags=list(dict.fromkeys(tags))[:15],
        credits=credits,
    )


def write_metadata(meta: Metadata, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "\n".join(
            [
                f"# {meta.title}",
                "",
                "## Tags",
                ", ".join(meta.tags),
                "",
                "## Description",
                "",
                "```",
                meta.description,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dest
