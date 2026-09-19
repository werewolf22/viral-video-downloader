"""Pick which few seconds of a long video are worth keeping.

Taking the first N seconds (what the original process.py did) reliably captures
intros, sponsor reads and channel bumpers - the least viral part of any video.
The payoff moment is almost always where the audio gets loudest and busiest, so
that is what we search for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import HighlightConfig
from .ffmpeg_utils import detect_scene_changes, extract_pcm, media_info
from .util import get_logger

log = get_logger("highlight")

WINDOW_SEC = 0.25  # analysis resolution


@dataclass
class Window:
    start: float
    duration: float
    method: str
    confidence: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


def _search_bounds(total: float, cfg: HighlightConfig) -> tuple[float, float, float]:
    """Return (earliest_start, latest_start, clip_duration) clamped to the media."""
    clip = min(cfg.clip_seconds, total)
    earliest = min(cfg.skip_intro_sec, max(total - clip, 0.0))
    latest = max(total - cfg.skip_outro_sec - clip, earliest)
    return earliest, latest, clip


def by_audio_energy(source: str | Path, total: float, cfg: HighlightConfig) -> Window:
    """Loudest sustained ``clip_seconds`` window, measured as smoothed RMS."""
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        log.warning("numpy not installed - falling back to a fixed-offset clip")
        return by_offset(total, cfg)

    earliest, latest, clip = _search_bounds(total, cfg)
    if latest <= earliest:
        return Window(start=earliest, duration=clip, method="whole")

    raw = extract_pcm(source, sample_rate=16_000)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        log.info("%s has no audio - using fixed offset", Path(source).name)
        return by_offset(total, cfg)

    frame = int(16_000 * WINDOW_SEC)
    usable = (samples.size // frame) * frame
    if usable == 0:
        return by_offset(total, cfg)

    frames = samples[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))

    # Rolling mean over a full clip length: rewards sustained energy (a laugh, a
    # crash, a crowd) rather than a single transient spike.
    span = max(int(clip / WINDOW_SEC), 1)
    if span > rms.size:
        return Window(start=earliest, duration=min(clip, total), method="whole")
    kernel = np.ones(span, dtype=np.float32) / span
    rolling = np.convolve(rms, kernel, mode="valid")

    lo = int(earliest / WINDOW_SEC)
    hi = min(int(latest / WINDOW_SEC) + 1, rolling.size)
    if hi <= lo:
        lo, hi = 0, rolling.size

    region = rolling[lo:hi]
    best = int(np.argmax(region)) + lo
    start = round(best * WINDOW_SEC, 3)

    baseline = float(np.mean(rolling)) or 1e-6
    confidence = float(region.max() / baseline)
    log.debug("%s: audio peak at %.1fs (%.2fx mean loudness)", Path(source).name, start, confidence)
    return Window(start=start, duration=clip, method="audio_energy", confidence=round(confidence, 3))


def by_scene_density(source: str | Path, total: float, cfg: HighlightConfig) -> Window:
    """Window containing the most cuts - a proxy for the fast-paced section."""
    earliest, latest, clip = _search_bounds(total, cfg)
    scenes = detect_scene_changes(source, cfg.scene_threshold)
    if not scenes:
        log.debug("%s: no scene changes detected", Path(source).name)
        return by_offset(total, cfg)

    best_start, best_count = earliest, -1
    step = 0.5
    position = earliest
    while position <= latest:
        count = sum(1 for t in scenes if position <= t < position + clip)
        if count > best_count:
            best_start, best_count = position, count
        position += step

    return Window(start=round(best_start, 3), duration=clip, method="scene", confidence=float(best_count))


def by_offset(total: float, cfg: HighlightConfig) -> Window:
    earliest, _, clip = _search_bounds(total, cfg)
    return Window(start=round(earliest, 3), duration=clip, method="start")


def find_highlight(source: str | Path, cfg: HighlightConfig, total: float | None = None) -> Window:
    """Dispatch to the configured strategy, degrading gracefully on failure."""
    total = total if total and total > 0 else media_info(source).duration

    if total <= cfg.clip_seconds:
        return Window(start=0.0, duration=total, method="whole")

    # If the caller supplied a suggested start, validate it inside the safe bounds
    # and use it as a soft preference. We still run the configured detector so the
    # final cut lands on an actual audio peak / scene cut near the LLM suggestion.
    suggested_start = getattr(cfg, "_suggested_start", None)

    strategies = {
        "audio_energy": lambda: by_audio_energy(source, total, cfg),
        "scene": lambda: by_scene_density(source, total, cfg),
        "start": lambda: by_offset(total, cfg),
    }
    strategy = strategies.get(cfg.method)
    if strategy is None:
        log.warning("Unknown highlight.method=%r, using audio_energy", cfg.method)
        strategy = strategies["audio_energy"]

    try:
        window = strategy()
    except Exception as exc:
        log.warning("Highlight detection failed for %s (%s) - using fixed offset", source, exc)
        window = by_offset(total, cfg)

    if suggested_start is not None and 0.0 <= suggested_start < total - cfg.clip_seconds:
        # Nudge the window so it starts as close as possible to the LLM suggestion
        # while keeping the full clip length inside the media.
        end = min(suggested_start + cfg.clip_seconds, total)
        window = Window(
            start=round(max(0.0, end - cfg.clip_seconds), 3),
            duration=round(min(cfg.clip_seconds, end), 3),
            method=f"{window.method}+llm_hint",
            confidence=window.confidence,
        )
    return window
