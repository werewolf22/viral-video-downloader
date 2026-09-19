"""Stage 4 - transcribe each clip into a valid, readable SRT.

Two things the original version got wrong:

* it wrote raw float seconds (``0.0 --> 4.2``) as timestamps, which no SRT
  parser accepts - ffmpeg/libass silently render nothing;
* it emitted whole Whisper segments, which are often 15+ words and unreadable
  as burned-in captions.

Here timestamps are ``HH:MM:SS,mmm`` and cues are re-chunked to a few words at a
time using word-level timings when the engine provides them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import CaptionConfig, Config, load_config
from .ffmpeg_utils import extract_audio_file
from .process import Clip, load_clips
from .util import StageError, get_logger, srt_timestamp, write_json

log = get_logger("subtitles")

_MODEL_CACHE: dict[tuple[str, str], object] = {}
MAX_CUE_SECONDS = 3.5
WORD_GAP_SECONDS = 0.7


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Cue:
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def _available_engine(preference: str) -> str:
    """Resolve 'auto' to whichever Whisper implementation is installed."""
    if preference != "auto":
        return preference

    from importlib.util import find_spec  # noqa: PLC0415

    if find_spec("faster_whisper"):
        return "faster-whisper"
    if find_spec("whisper"):
        return "whisper"
    return "none"


def _load_model(engine: str, size: str):
    key = (engine, size)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    log.info("Loading %s model %r (first run downloads weights)", engine, size)
    if engine == "faster-whisper":
        from faster_whisper import WhisperModel  # noqa: PLC0415

        # int8 on CPU is ~4x faster than float32 with negligible quality loss here.
        model = WhisperModel(size, device="cpu", compute_type="int8")
    elif engine == "whisper":
        import whisper  # noqa: PLC0415

        model = whisper.load_model(size)
    else:
        raise StageError(f"Unknown caption engine {engine!r}")

    _MODEL_CACHE[key] = model
    return model


def transcribe_words(audio: Path, cfg: CaptionConfig) -> list[Word]:
    engine = _available_engine(cfg.engine)
    model = _load_model(engine, cfg.model)
    words: list[Word] = []

    if engine == "faster-whisper":
        segments, _info = model.transcribe(
            str(audio), language=cfg.language, word_timestamps=True, vad_filter=True
        )
        for segment in segments:
            if segment.words:
                words += [Word(w.start, w.end, w.word.strip()) for w in segment.words if w.word.strip()]
            elif segment.text.strip():
                words.append(Word(segment.start, segment.end, segment.text.strip()))
    else:
        result = model.transcribe(str(audio), language=cfg.language, word_timestamps=True)
        for segment in result.get("segments", []):
            segment_words = segment.get("words") or []
            if segment_words:
                words += [
                    Word(float(w["start"]), float(w["end"]), str(w["word"]).strip())
                    for w in segment_words
                    if str(w.get("word", "")).strip()
                ]
            elif segment.get("text", "").strip():
                words.append(Word(float(segment["start"]), float(segment["end"]), segment["text"].strip()))

    return words


# ---------------------------------------------------------------------------
# Cue construction
# ---------------------------------------------------------------------------

def group_into_cues(words: list[Word], cfg: CaptionConfig) -> list[Cue]:
    """Pack words into short cues, breaking on length, duration or a pause."""
    limit = cfg.max_chars_per_line * cfg.max_lines
    cues: list[Cue] = []
    buffer: list[Word] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(w.text for w in buffer).strip()
        if text:
            cues.append(Cue(start=buffer[0].start, end=max(buffer[-1].end, buffer[0].start + 0.4), text=text))
        buffer.clear()

    for word in words:
        if buffer:
            pending = len(" ".join(w.text for w in buffer)) + 1 + len(word.text)
            gap = word.start - buffer[-1].end
            span = word.end - buffer[0].start
            if pending > limit or gap > WORD_GAP_SECONDS or span > MAX_CUE_SECONDS:
                flush()
        buffer.append(word)
    flush()

    # Never let one cue overlap the next; libass renders overlaps stacked.
    for current, following in zip(cues, cues[1:]):
        current.end = min(current.end, following.start - 0.01)
    return [c for c in cues if c.end > c.start]


def wrap_lines(text: str, max_chars: int, max_lines: int) -> str:
    """Greedy word wrap so captions never run off a 9:16 frame."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines[:max_lines])


def write_srt(cues: list[Cue], dest: Path, cfg: CaptionConfig) -> Path:
    """Portable sidecar track - upload it as a caption file, or burn it in."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for index, cue in enumerate(cues, 1):
        body = wrap_lines(cue.text.upper(), cfg.max_chars_per_line, cfg.max_lines)
        blocks.append(
            f"{index}\n{srt_timestamp(cue.start)} --> {srt_timestamp(cue.end)}\n{body}\n"
        )
    dest.write_text("\n".join(blocks), encoding="utf-8")
    return dest


def _ass_timestamp(seconds: float) -> str:
    """ASS wants ``H:MM:SS.cc`` with centisecond precision."""
    seconds = max(seconds, 0.0)
    centis = int(round(seconds * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def write_ass(cues: list[Cue], dest: Path, cfg: CaptionConfig, width: int, height: int) -> Path:
    """Styled subtitle file used for burn-in.

    Written instead of relying on ``force_style`` over an SRT: ffmpeg's SRT
    converter fixes the ASS canvas at 384x288, so a FontSize/MarginV chosen for
    a 1080x1920 frame gets scaled by 6.7x and shoved off the top of the picture.
    Declaring PlayResX/PlayResY here makes both values honest pixels.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    style = ",".join(
        str(v) for v in [
            "Viral", cfg.font, cfg.font_size,
            cfg.primary_colour, "&H000000FF", cfg.outline_colour, "&H80000000",
            -1, 0, 0, 0,          # Bold, Italic, Underline, StrikeOut
            100, 100, 0, 0,       # ScaleX, ScaleY, Spacing, Angle
            1, cfg.outline, 2,    # BorderStyle, Outline, Shadow
            2,                    # Alignment: bottom-centre
            int(width * 0.07), int(width * 0.07), cfg.margin_v,
            1,
        ]
    )

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",  # honour our own line breaks, never re-wrap
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
        " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
        " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: {style}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for cue in cues:
        body = wrap_lines(cue.text.upper(), cfg.max_chars_per_line, cfg.max_lines)
        # Braces open an ASS override block; transcripts must not be able to inject one.
        body = body.replace("{", "(").replace("}", ")").replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(cue.start)},{_ass_timestamp(cue.end)},Viral,,0,0,0,,{body}"
        )

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def subtitle_one(
    clip_path: Path, cfg: Config, *, width: int | None = None, height: int | None = None
) -> dict[str, str] | None:
    """Transcribe any video file, returning the paths of its .srt and .ass tracks."""
    srt_path = cfg.path("subtitles") / f"{clip_path.stem}.srt"
    ass_path = cfg.path("subtitles") / f"{clip_path.stem}.ass"

    if srt_path.exists() and srt_path.stat().st_size > 0 and ass_path.exists():
        log.info("Cached  %s", srt_path.name)
        return {"srt": str(srt_path), "ass": str(ass_path)}

    audio = cfg.path("subtitles") / f"{clip_path.stem}.wav"
    try:
        extract_audio_file(clip_path, audio)
        words = transcribe_words(audio, cfg.captions)
    except Exception as exc:
        log.error("Transcription failed for %s: %s", clip_path.name, exc)
        return None
    finally:
        audio.unlink(missing_ok=True)

    cues = group_into_cues(words, cfg.captions)
    if not cues:
        log.info("No speech detected in %s", clip_path.name)
        return None

    render_w = width if width is not None else cfg.render.width
    render_h = height if height is not None else cfg.render.height
    write_srt(cues, srt_path, cfg.captions)
    write_ass(cues, ass_path, cfg.captions, render_w, render_h)
    log.info("Captioned %-28s %d cues", clip_path.name, len(cues))
    return {"srt": str(srt_path), "ass": str(ass_path)}


def subtitle_one_clip(clip: Clip, cfg: Config) -> dict[str, str] | None:
    """Backward-compatible wrapper for the old Clip API."""
    return subtitle_one(Path(clip.path), cfg)


def subtitle_assets(
    cfg: Config | None = None, assets: list[Asset] | None = None
) -> dict[str, dict[str, str]]:
    """Transcribe raw downloaded assets and write a subtitles.json-style map."""
    from .fetch import Asset, load_assets  # noqa: PLC0415

    cfg = cfg or load_config()
    cfg.make_dirs()
    assets = assets or load_assets(cfg)

    if not cfg.captions.enabled:
        log.info("Captions disabled in config - skipping")
        return {}

    engine = _available_engine(cfg.captions.engine)
    if engine == "none":
        log.warning(
            "No Whisper implementation installed - continuing without captions. "
            "Install faster-whisper (recommended) or openai-whisper to enable them."
        )
        return {}

    mapping: dict[str, dict[str, str]] = {}
    for asset in assets:
        tracks = subtitle_one(Path(asset.path), cfg)
        if tracks:
            mapping[asset.video_id] = tracks

    write_json(cfg.path("workspace") / "subtitles.json", mapping)
    log.info("Generated subtitles for %d/%d videos", len(mapping), len(assets))
    return mapping


def subtitles(cfg: Config | None = None, clips: list[Clip] | None = None) -> dict[str, dict[str, str]]:
    """Original stage entry point for processed clips."""
    cfg = cfg or load_config()
    cfg.make_dirs()
    clips = clips or load_clips(cfg)

    if not cfg.captions.enabled:
        log.info("Captions disabled in config - skipping")
        return {}

    engine = _available_engine(cfg.captions.engine)
    if engine == "none":
        log.warning(
            "No Whisper implementation installed - continuing without captions. "
            "Install faster-whisper (recommended) or openai-whisper to enable them."
        )
        return {}

    mapping: dict[str, dict[str, str]] = {}
    for clip in clips:
        tracks = subtitle_one_clip(clip, cfg)
        if tracks:
            mapping[clip.video_id] = tracks

    write_json(cfg.path("workspace") / "subtitles.json", mapping)
    log.info("Generated subtitles for %d/%d clips", len(mapping), len(clips))
    return mapping


if __name__ == "__main__":
    subtitles()
