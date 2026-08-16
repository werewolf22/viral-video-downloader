"""Typed configuration loaded from ``config.yaml`` + environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .util import ensure_dir, get_logger

log = get_logger("config")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class DiscoveryConfig:
    # "youtube_api" needs YOUTUBE_API_KEY; "ytdlp_search" works with no key;
    # "file" reads urls from `sources_file`.
    source: str = "youtube_api"
    region: str = "US"
    category_id: str | None = None
    search_queries: list[str] = field(
        default_factory=lambda: ["funny fails", "satisfying moments", "unexpected moments"]
    )
    sources_file: str = "sources.txt"
    max_candidates: int = 50
    max_results: int = 6
    min_views: int = 100_000
    min_duration_sec: int = 15
    max_duration_sec: int = 900
    published_within_hours: int = 168
    # "any" or "creativeCommon". creativeCommon restricts results to CC-BY videos,
    # which are the ones you may legally remix and re-upload with attribution.
    license_filter: str = "any"
    blocked_channels: list[str] = field(default_factory=list)
    blocked_keywords: list[str] = field(default_factory=list)


@dataclass
class DownloadConfig:
    # yt-dlp format selector; caps at 1080p so downloads stay a sane size.
    format: str = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b"
    max_filesize_mb: int = 500
    retries: int = 3
    cookies_from_browser: str | None = None
    rate_limit: str | None = None


@dataclass
class HighlightConfig:
    clip_seconds: float = 8.0
    # "audio_energy" picks the loudest sustained window, "scene" the densest cut
    # region, "start" just takes the opening (fastest, dumbest).
    method: str = "audio_energy"
    skip_intro_sec: float = 3.0
    skip_outro_sec: float = 3.0
    scene_threshold: float = 0.35


@dataclass
class RenderConfig:
    width: int = 1080
    height: int = 1920
    fps: int = 30
    # "blur_pad" keeps the whole frame over a blurred backdrop, "crop" fills by
    # center-cropping, "pad" uses flat bars.
    fit: str = "blur_pad"
    loudness_lufs: float = -14.0
    crf: int = 20
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    transition: str = "none"  # "none" | "fade"
    transition_seconds: float = 0.35


@dataclass
class CaptionConfig:
    enabled: bool = True
    engine: str = "auto"  # "auto" | "faster-whisper" | "whisper" | "none"
    model: str = "base"
    language: str | None = None
    max_chars_per_line: int = 24
    max_lines: int = 2
    font: str = "DejaVu Sans"
    font_size: int = 20
    margin_v: int = 260
    primary_colour: str = "&H00FFFFFF"
    outline_colour: str = "&H00000000"
    outline: int = 3


@dataclass
class BrandingConfig:
    title_card_text: str = "TOP MOMENTS\nOF THE WEEK"
    title_card_seconds: float = 2.0
    clip_label: bool = True  # burn "#1", "#2", ... in the corner
    watermark_text: str | None = None
    outro_credits: bool = True
    outro_seconds: float = 3.5
    font_file: str | None = None  # auto-detected when null
    accent_colour: str = "#FF2D55"


@dataclass
class PathsConfig:
    workspace: str = "workspace"
    downloads: str = "workspace/downloads"
    clips: str = "workspace/clips"
    subtitles: str = "workspace/subtitles"
    renders: str = "workspace/renders"
    output: str = "output"


@dataclass
class Config:
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    branding: BrandingConfig = field(default_factory=BrandingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    youtube_api_key: str | None = None
    root: Path = field(default_factory=lambda: PROJECT_ROOT)

    # -- derived paths -----------------------------------------------------
    def path(self, name: str) -> Path:
        """Absolute path for one of the configured workspace directories."""
        raw = getattr(self.paths, name)
        p = Path(raw)
        return p if p.is_absolute() else self.root / p

    def make_dirs(self) -> None:
        for f in fields(self.paths):
            ensure_dir(self.path(f.name))

    @property
    def state_file(self) -> Path:
        return self.path("workspace") / "state.json"


def _merge(instance: Any, data: dict[str, Any], trail: str = "") -> None:
    """Apply a dict of overrides onto a dataclass instance, recursing into nested ones."""
    known = {f.name: f for f in fields(instance)}
    for key, value in data.items():
        if key not in known:
            log.warning("Ignoring unknown config key: %s%s", trail, key)
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, trail=f"{trail}{key}.")
        else:
            setattr(instance, key, value)


def _load_dotenv(root: Path) -> None:
    """Minimal .env reader so the project has no hard dependency on python-dotenv."""
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Build a Config from defaults, then ``config.yaml``, then the environment."""
    cfg = Config()
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if config_path.exists():
        try:
            import yaml  # noqa: PLC0415 - optional dependency, defaults work without it
        except ImportError:
            log.warning("PyYAML not installed - using built-in defaults, ignoring %s", config_path)
        else:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            _merge(cfg, data)
            log.debug("Loaded config from %s", config_path)

    _load_dotenv(cfg.root)
    cfg.youtube_api_key = os.environ.get("YOUTUBE_API_KEY") or cfg.youtube_api_key
    return cfg
