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
    # When enabled, the LLM writes fresh search queries for every run instead of
    # replaying the static list below. This is the fastest way to stop the pipeline
    # from returning the same videos every time.
    llm_query_enabled: bool = False
    llm_query_theme: str = "viral, surprising, funny, satisfying or caught-on-camera moments trending this week"
    llm_query_count: int = 10
    llm_query_temperature: float = 0.7


@dataclass
class DownloadConfig:
    # Default yt-dlp format selector. The pipeline can also ask an LLM to pick
    # a format from the available list when the default selector is rejected.
    format: str = "bestvideo[ext=mp4][vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/b[height<=1080]/b"
    max_filesize_mb: int = 500
    retries: int = 3
    cookies_from_browser: str | None = None
    rate_limit: str | None = None
    # If true and yt-dlp reports "Requested format is not available", fetch the
    # format list, feeds it to the LLM, and downloads the format the LLM picks.
    llm_format_fallback: bool = True


@dataclass
class TwitchConfig:
    # Twitch Helix API credentials are read from TWITCH_CLIENT_ID and
    # TWITCH_CLIENT_SECRET in .env. discovery.source: twitch uses these to
    # fetch trending clips by game/category and view count.
    enabled: bool = False
    game_name: str | None = None
    period: str = "week"  # day | week | month | all
    language: str | None = None  # e.g. "en"
    # Future manual source; not wired in the first API-only pass.
    sources_file: str = "twitch_sources.txt"


@dataclass
class MusicVideoConfig:
    # Music VIDEO clip mode. discovery.source: music_video uses these settings
    # to surface viral clips that contain music + video (Shorts, music videos,
    # trending songs with visuals). Results are always videos, never audio-only.
    enabled: bool = False
    genres: list[str] = field(default_factory=lambda: ["pop", "hip hop", "electronic", "rock"])
    queries: list[str] = field(default_factory=lambda: [
        "viral music video",
        "trending song video",
        "popular music clip",
    ])
    use_shorts: bool = True


@dataclass
class InstagramConfig:
    # Instagram Reels/clip discovery. The API has no free viral endpoint, so
    # this supports a manual URL file or a third-party backend (Apify).
    # Everything here targets clips WITH music/video, never static photos.
    enabled: bool = False
    backend: str = "manual"  # manual | apify
    sources_file: str = "instagram_sources.txt"
    api_key: str | None = None
    base_url: str | None = "https://api.apify.com/v2"
    actor_id: str | None = None  # e.g. "apify/instagram-scraper"
    # Music-video hashtags to scrape, e.g. ["viralmusic", "musicvideo"].
    hashtags: list[str] = field(default_factory=lambda: ["viralmusic"])
    # Free-form search terms the Apify actor may support.
    queries: list[str] = field(default_factory=list)
    max_results: int = 50


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
    max_chars_per_line: int = 22
    max_lines: int = 2
    font: str = "DejaVu Sans"
    # font_size, margin_v and outline are pixels at the configured render size.
    font_size: int = 72
    margin_v: int = 280
    primary_colour: str = "&H00FFFFFF"  # ASS colours are &HAABBGGRR
    outline_colour: str = "&H00000000"
    outline: int = 6


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
class CuratorConfig:
    # Optional LLM curation between discovery and fetch.
    enabled: bool = False
    provider: str = "ollama"  # "ollama" | "openai" | any OpenAI-compatible
    base_url: str = "http://localhost:11434/v1"
    model: str = "deepseek-v4-flash:cloud"
    api_key: str | None = None
    max_candidates: int = 20      # how many discovery results the LLM reviews
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120


@dataclass
class PathsConfig:
    workspace: str = "workspace"
    downloads: str = "workspace/downloads"
    twitch_downloads: str = "workspace/downloads_twitch"
    youtube_music_downloads: str = "workspace/downloads_music_video"
    instagram_downloads: str = "workspace/downloads_instagram"
    clips: str = "workspace/clips"
    subtitles: str = "workspace/subtitles"
    renders: str = "workspace/renders"
    output: str = "output"


@dataclass
class Config:
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    twitch: TwitchConfig = field(default_factory=TwitchConfig)
    music_video: MusicVideoConfig = field(default_factory=MusicVideoConfig)
    instagram: InstagramConfig = field(default_factory=InstagramConfig)
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    branding: BrandingConfig = field(default_factory=BrandingConfig)
    curator: CuratorConfig = field(default_factory=CuratorConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    youtube_api_key: str | None = None
    twitch_client_id: str | None = None
    twitch_client_secret: str | None = None
    instagram_api_key: str | None = None
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
    cfg.twitch_client_id = os.environ.get("TWITCH_CLIENT_ID") or cfg.twitch_client_id
    cfg.twitch_client_secret = os.environ.get("TWITCH_CLIENT_SECRET") or cfg.twitch_client_secret
    cfg.instagram_api_key = os.environ.get("INSTAGRAM_API_KEY") or cfg.instagram_api_key
    return cfg
