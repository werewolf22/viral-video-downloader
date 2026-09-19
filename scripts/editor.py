"""Stage 5 - assemble the final compilation.

Structure of the output:

    [title card] -> [clip 1 + captions] -> ... -> [clip N] -> [credits card]

Each clip gets its overlays burned in individually, then the parts are joined
with the concat demuxer. Because every part was encoded to the identical spec in
the process stage, the join is a stream copy: fast, and lossless.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from .config import Config, load_config
from .ffmpeg_utils import (
    apply_overlays,
    build_drawtext,
    build_subtitles_filter,
    concat,
    find_font_file,
    media_info,
    render_card,
    require_ffmpeg,
)
from .metadata import build_metadata, write_metadata
from .process import Clip, load_clips
from .util import StageError, get_logger, human_duration, read_json

log = get_logger("editor")


def _subtitle_track(entry: object) -> Path | None:
    """Pick the styled .ass track if the subtitles stage produced one."""
    if isinstance(entry, dict):
        chosen = entry.get("ass") or entry.get("srt")
    elif isinstance(entry, str):
        chosen = entry
    else:
        return None
    if not chosen:
        return None
    path = Path(chosen)
    return path if path.exists() and path.stat().st_size > 0 else None


def _clip_overlays(index: int, cfg: Config, track: Path | None, tmp: Path) -> list[str]:
    """Filters burned onto a single clip: captions, position badge, watermark."""
    font_file = find_font_file(cfg.branding.font_file)
    filters: list[str] = []

    if track:
        filters.append(build_subtitles_filter(track, cfg.captions, cfg.render.height))

    if cfg.branding.clip_label:
        label_file = tmp / f"label_{index}.txt"
        label_file.write_text(f"#{index}", encoding="utf-8")
        filters.append(
            build_drawtext(
                label_file, font_file,
                x=f"{int(cfg.render.width * 0.055)}",
                y=f"{int(cfg.render.height * 0.055)}",
                size=int(cfg.render.height * 0.045),
                colour="white",
                box=True,
                box_colour="black@0.55",
                # Fades out after the first couple of seconds so it never covers the action.
                enable="lt(t,2.2)",
            )
        )

    if cfg.branding.watermark_text:
        mark_file = tmp / f"mark_{index}.txt"
        mark_file.write_text(cfg.branding.watermark_text, encoding="utf-8")
        filters.append(
            build_drawtext(
                mark_file, font_file,
                x="(w-text_w)/2",
                y=f"h-{int(cfg.render.height * 0.055)}-text_h",
                size=int(cfg.render.height * 0.022),
                colour="white@0.75",
            )
        )

    return filters


def _credits_text(clips: list[Clip], limit: int = 8) -> str:
    seen: list[str] = []
    for clip in clips:
        if clip.channel and clip.channel not in seen:
            seen.append(clip.channel)
    lines = seen[:limit]
    if len(seen) > limit:
        lines.append(f"+ {len(seen) - limit} more")
    return "\n".join(lines)


def build(cfg: Config | None = None, clips: list[Clip] | None = None) -> Path:
    cfg = cfg or load_config()
    cfg.make_dirs()
    require_ffmpeg()
    clips = clips or load_clips(cfg)

    if not clips:
        raise StageError("No clips available to edit")

    subtitle_map = read_json(cfg.path("workspace") / "subtitles.json", default={}) or {}
    renders_dir = cfg.path("renders")
    parts: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)

        if cfg.branding.title_card_text and cfg.branding.title_card_seconds > 0:
            card = render_card(
                renders_dir / "00_title.mp4",
                text=cfg.branding.title_card_text,
                duration=cfg.branding.title_card_seconds,
                render=cfg.render,
                font_file=find_font_file(cfg.branding.font_file),
                subtitle=datetime.now().strftime("%B %Y"),
            )
            parts.append(card)

        for index, clip in enumerate(clips, 1):
            source = Path(clip.path)
            if not source.exists():
                log.error("Clip missing on disk, skipping: %s", source)
                continue

            track = _subtitle_track(subtitle_map.get(clip.video_id))
            filters = _clip_overlays(index, cfg, track, tmp)
            dest = renders_dir / f"{index:02d}_{clip.video_id}_final.mp4"

            try:
                apply_overlays(source, dest, video_filters=filters, render=cfg.render)
            except StageError as exc:
                log.error("Overlay pass failed for clip %d: %s", index, exc)
                continue

            log.info("Rendered clip %02d  %-38.38s  %s", index, clip.title, clip.channel)
            parts.append(dest)

        if cfg.branding.outro_credits:
            outro = render_card(
                renders_dir / "99_credits.mp4",
                text="CREDITS",
                duration=cfg.branding.outro_seconds,
                render=cfg.render,
                font_file=find_font_file(cfg.branding.font_file),
                background="#111111",
                subtitle=_credits_text(clips),
            )
            parts.append(outro)

    if not parts:
        raise StageError("Nothing rendered - all clips failed the overlay pass")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = cfg.path("output") / f"compilation-{stamp}.mp4"
    concat(parts, output)

    info = media_info(output)
    log.info(
        "Final video: %s  (%s, %dx%d, %.1f MB)",
        output, human_duration(info.duration), info.width, info.height,
        output.stat().st_size / 1_048_576,
    )

    meta = build_metadata(clips, cfg, duration=info.duration)
    write_metadata(meta, output.with_suffix(".md"))
    log.info("Upload metadata + credits: %s", output.with_suffix(".md"))
    return output


if __name__ == "__main__":
    build()
