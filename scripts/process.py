"""Stage 3 - cut each download to its highlight and normalize it.

Concatenation is only cheap when every part shares a codec, resolution, frame
rate, pixel format and audio layout, so this stage forces all of them to the
render spec. It also fixes the original bug where ``subclip(0, 8)`` crashed on
any source shorter than eight seconds.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config, HighlightConfig, load_config
from .curator import CuratedCandidate, load_curated_candidates  # noqa: F401 - used in isinstance
from .fetch import Asset, load_assets
from .ffmpeg_utils import media_info, render_segment, require_ffmpeg
from .highlight import find_highlight
from .util import StageError, get_logger, human_duration, read_json, write_json


def _copy_highlight_cfg(cfg: HighlightConfig) -> HighlightConfig:
    """Deep-copy a HighlightConfig so the LLM hint is per-asset, not global."""
    return HighlightConfig(
        clip_seconds=cfg.clip_seconds,
        method=cfg.method,
        skip_intro_sec=cfg.skip_intro_sec,
        skip_outro_sec=cfg.skip_outro_sec,
        scene_threshold=cfg.scene_threshold,
    )

log = get_logger("process")


@dataclass
class Clip:
    video_id: str
    path: str
    title: str
    channel: str
    channel_url: str
    source_url: str
    license: str
    platform: str = "youtube"
    start: float = 0.0
    duration: float = 0.0
    method: str = "start"
    score: float = 0.0


def process_one(asset: Asset, cfg: Config, index: int) -> Clip | None:
    source = Path(asset.path)
    if not source.exists():
        log.error("Missing download: %s", source)
        return None

    try:
        info = media_info(source)
    except StageError as exc:
        log.error("Skipping unreadable file %s: %s", source.name, exc)
        return None

    highlight_cfg = cfg.highlight
    if asset.llm_suggested_start > 0:
        highlight_cfg = _copy_highlight_cfg(highlight_cfg)
        setattr(highlight_cfg, "_suggested_start", asset.llm_suggested_start)
        log.debug("Using LLM suggested start %.1fs for %s", asset.llm_suggested_start, asset.video_id)

    window = find_highlight(source, highlight_cfg, total=info.duration)
    dest = cfg.path("clips") / f"{index:02d}_{asset.video_id}.mp4"

    log.info(
        "Clip %02d  %-40.40s  %s -> %s  (%s)",
        index, asset.title,
        human_duration(window.start), human_duration(window.end), window.method,
    )

    try:
        render_segment(
            source, dest,
            start=window.start,
            duration=window.duration,
            render=cfg.render,
            has_audio=info.has_audio,
            fade=cfg.render.transition == "fade",
        )
    except StageError as exc:
        log.error("Failed to render clip from %s: %s", source.name, exc)
        return None

    return Clip(
        video_id=asset.video_id,
        path=str(dest),
        title=asset.title,
        channel=asset.channel,
        channel_url=asset.channel_url,
        source_url=asset.source_url,
        license=asset.license,
        platform=asset.platform,
        start=window.start,
        duration=window.duration,
        method=window.method,
        score=asset.score,
    )


def process(cfg: Config | None = None, assets: list[Asset] | None = None) -> list[Clip]:
    cfg = cfg or load_config()
    cfg.make_dirs()
    require_ffmpeg()
    assets = assets or load_assets(cfg)

    log.info(
        "Normalizing to %dx%d @ %dfps (fit=%s, %.1fs per clip)",
        cfg.render.width, cfg.render.height, cfg.render.fps,
        cfg.render.fit, cfg.highlight.clip_seconds,
    )

    clips: list[Clip] = []
    for index, asset in enumerate(assets, 1):
        clip = process_one(asset, cfg, index)
        if clip:
            clips.append(clip)

    if not clips:
        raise StageError("No clips were produced - every source failed to process")

    out = cfg.path("workspace") / "clips.json"
    write_json(out, [asdict(c) for c in clips])
    total = sum(c.duration for c in clips)
    log.info("Produced %d clips totalling %s", len(clips), human_duration(total))
    return clips


def load_clips(cfg: Config) -> list[Clip]:
    data = read_json(cfg.path("workspace") / "clips.json")
    if not data:
        raise StageError("clips.json not found or empty - run the process stage first")
    return [Clip(**item) for item in data]


if __name__ == "__main__":
    process()
