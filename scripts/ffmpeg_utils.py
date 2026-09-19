"""Thin, explicit wrappers around ffmpeg/ffprobe.

Every video operation in this project goes through ffmpeg directly rather than a
Python editing library: it is faster, has a stable CLI contract, and avoids the
MoviePy 1.x/2.x API split that silently breaks pipelines on upgrade.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .util import StageError, get_logger

log = get_logger("ffmpeg")

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arialbd.ttf",
]


@dataclass
class MediaInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def require_ffmpeg() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise StageError(
            f"Missing required binaries: {', '.join(missing)}. "
            "Install ffmpeg (e.g. `sudo apt install ffmpeg`) and retry."
        )


def find_font_file(override: str | None = None) -> str:
    if override and Path(override).exists():
        return override
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise StageError(
        "No usable TTF font found for text overlays. Set branding.font_file in config.yaml."
    )


def run(cmd: list[str], desc: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, raising StageError with the tail of stderr on failure."""
    log.debug("$ %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise StageError(f"{desc}: timed out after {timeout}s") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise StageError(f"{desc} failed (exit {proc.returncode}):\n{tail}")
    return proc


def ffprobe(path: str | Path) -> dict:
    proc = run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        desc=f"ffprobe {Path(path).name}",
    )
    return json.loads(proc.stdout)


def _parse_fps(rate: str | None) -> float:
    if not rate or "/" not in rate:
        return 30.0
    num, _, den = rate.partition("/")
    try:
        num_f, den_f = float(num), float(den)
    except ValueError:
        return 30.0
    return num_f / den_f if den_f else 30.0


def media_info(path: str | Path) -> MediaInfo:
    data = ffprobe(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise StageError(f"{path} has no video stream")

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise StageError(f"Could not determine duration of {path}")

    return MediaInfo(
        path=Path(path),
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------

def escape_filter_path(path: str | Path) -> str:
    """Escape a path for use inside a filtergraph option value."""
    return (
        str(path)
        .replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def build_fit_filter(width: int, height: int, fps: int, fit: str) -> str:
    """Filtergraph that reframes any input to exactly width x height."""
    tail = f"setsar=1,fps={fps},format=yuv420p"

    if fit == "crop":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},{tail}"
        )
    if fit == "pad":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,{tail}"
        )
    if fit != "blur_pad":
        log.warning("Unknown render.fit=%r, falling back to blur_pad", fit)

    # Blurred, darkened copy of the frame behind a fully visible foreground.
    return (
        "split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=luma_radius=40:luma_power=2,eq=brightness=-0.15[bgb];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,{tail}"
    )


SRT_ASS_CANVAS_HEIGHT = 288  # ffmpeg's srt->ass converter hard-codes PlayRes 384x288


def build_subtitles_filter(path: str | Path, captions, frame_height: int) -> str:
    """Burn-in filter for a subtitle track.

    ``.ass`` files carry their own PlayRes and style, so they render exactly as
    written. ``.srt`` has no canvas of its own: ffmpeg converts it on a 384x288
    script, and libass then scales that up to the frame. Font size and margin
    therefore have to be divided down by the same factor, or a 260px margin
    becomes a 1733px one and the captions leave the picture.
    """
    path = Path(path)
    escaped = escape_filter_path(path)
    if path.suffix.lower() in (".ass", ".ssa"):
        return f"subtitles='{escaped}'"

    scale = frame_height / SRT_ASS_CANVAS_HEIGHT
    style = ",".join(
        [
            f"FontName={captions.font}",
            f"FontSize={max(int(captions.font_size / scale), 8)}",
            f"PrimaryColour={captions.primary_colour}",
            f"OutlineColour={captions.outline_colour}",
            "BorderStyle=1",
            f"Outline={max(int(captions.outline / scale), 1)}",
            "Shadow=1",
            "Bold=1",
            "Alignment=2",
            f"MarginV={int(captions.margin_v / scale)}",
        ]
    )
    return f"subtitles='{escaped}':force_style='{style}'"


def build_drawtext(
    textfile: str | Path,
    font_file: str,
    *,
    x: str,
    y: str,
    size: int,
    colour: str = "white",
    box: bool = False,
    box_colour: str = "black@0.5",
    line_spacing: int = 12,
    enable: str | None = None,
) -> str:
    """drawtext reading from a file, so caption text needs no shell/filter escaping."""
    parts = [
        f"textfile='{escape_filter_path(textfile)}'",
        f"fontfile='{escape_filter_path(font_file)}'",
        "expansion=none",
        f"fontcolor={colour}",
        f"fontsize={size}",
        f"line_spacing={line_spacing}",
        f"x={x}",
        f"y={y}",
    ]
    if box:
        parts += ["box=1", f"boxcolor={box_colour}", "boxborderw=24"]
    if enable:
        parts.append(f"enable='{enable}'")
    return "drawtext=" + ":".join(parts)


# ---------------------------------------------------------------------------
# Encoding operations
# ---------------------------------------------------------------------------

def video_encode_args(render) -> list[str]:
    return [
        "-c:v", "libx264",
        "-preset", render.preset,
        "-crf", str(render.crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-movflags", "+faststart",
    ]


def encode_args(render) -> list[str]:
    """Encoder settings shared by every part, so concat can stream-copy later."""
    return [
        *video_encode_args(render),
        "-c:a", "aac",
        "-b:a", render.audio_bitrate,
        "-ar", "48000",
        "-ac", "2",
    ]


def render_segment(
    source: str | Path,
    dest: str | Path,
    *,
    start: float,
    duration: float,
    render,
    has_audio: bool,
    extra_video_filters: list[str] | None = None,
    fade: bool = False,
) -> Path:
    """Trim ``source`` to [start, start+duration] and normalize it to the render spec."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    video_chain = build_fit_filter(render.width, render.height, render.fps, render.fit)
    for extra in extra_video_filters or []:
        video_chain += f",{extra}"

    if fade and duration > 2 * render.transition_seconds:
        d = render.transition_seconds
        video_chain += f",fade=t=in:st=0:d={d},fade=t=out:st={duration - d:.3f}:d={d}"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(source)]

    audio_filter = (
        f"loudnorm=I={render.loudness_lufs}:TP=-1.5:LRA=11,"
        "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    )
    if has_audio:
        cmd += ["-filter_complex", f"[0:v]{video_chain}[v];[0:a]{audio_filter}[a]",
                "-map", "[v]", "-map", "[a]"]
    else:
        # Silent source: synthesise a silent track so every clip has the same layout
        # and the concat demuxer does not choke.
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-filter_complex", f"[0:v]{video_chain}[v]",
                "-map", "[v]", "-map", "1:a"]

    cmd += ["-t", f"{duration:.3f}", *encode_args(render), str(dest)]
    run(cmd, desc=f"render segment {dest.name}")
    return dest


def render_card(
    dest: str | Path,
    *,
    text: str,
    duration: float,
    render,
    font_file: str,
    background: str = "black",
    font_size: int | None = None,
    subtitle: str | None = None,
) -> Path:
    """Render a solid-colour title/outro card with centred text."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = font_size or int(render.height * 0.055)

    with tempfile.TemporaryDirectory() as tmp:
        main_txt = Path(tmp) / "main.txt"
        main_txt.write_text(text, encoding="utf-8")

        filters = [
            build_drawtext(
                main_txt, font_file,
                x="(w-text_w)/2",
                y="(h-text_h)/2" if not subtitle else "(h-text_h)/2-h*0.06",
                size=size,
                line_spacing=int(size * 0.35),
            )
        ]
        if subtitle:
            sub_txt = Path(tmp) / "sub.txt"
            sub_txt.write_text(subtitle, encoding="utf-8")
            filters.append(
                build_drawtext(
                    sub_txt, font_file,
                    x="(w-text_w)/2", y="h/2+h*0.04",
                    size=int(size * 0.5), colour="#DDDDDD",
                    line_spacing=int(size * 0.25),
                )
            )
        filters.append("fade=t=in:st=0:d=0.3")
        filters.append(f"fade=t=out:st={max(duration - 0.3, 0):.3f}:d=0.3")

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi",
            "-i", f"color=c={background}:s={render.width}x{render.height}:d={duration}:r={render.fps}",
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-vf", ",".join(filters),
            "-t", f"{duration:.3f}",
            *encode_args(render),
            str(dest),
        ]
        run(cmd, desc=f"render card {dest.name}")
    return dest


def apply_overlays(
    source: str | Path,
    dest: str | Path,
    *,
    video_filters: list[str],
    render,
) -> Path:
    """Re-encode an already-normalized clip with captions/labels burned in.

    Audio is stream-copied: it was loudness-normalized in the process stage and
    nothing here touches it.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    filters = [f for f in video_filters if f]

    if not filters:
        shutil.copyfile(source, dest)
        return dest

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-vf", ",".join(filters),
        *video_encode_args(render),
        "-c:a", "copy",
        str(dest),
    ]
    run(cmd, desc=f"overlay pass {dest.name}")
    return dest


def concat(parts: list[Path], dest: str | Path) -> Path:
    """Join identically-encoded parts without re-encoding."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        raise StageError("Nothing to concatenate")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        for part in parts:
            escaped = str(Path(part).resolve()).replace("'", "'\\''")
            fh.write(f"file '{escaped}'\n")
        list_path = fh.name

    try:
        run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", list_path,
                "-fflags", "+genpts", "-c", "copy",
                "-movflags", "+faststart", str(dest),
            ],
            desc=f"concat {len(parts)} parts",
        )
    finally:
        Path(list_path).unlink(missing_ok=True)
    return dest


def extract_pcm(source: str | Path, sample_rate: int = 16_000) -> bytes:
    """Mono 16-bit PCM of the whole file, for loudness analysis."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
        ],
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.decode("utf-8", "replace").strip().splitlines()[-10:])
        raise StageError(f"PCM extraction failed for {source}:\n{tail}")
    return proc.stdout


def extract_audio_file(source: str | Path, dest: str | Path, sample_rate: int = 16_000) -> Path:
    """16 kHz mono WAV — the format Whisper wants, without a video decode per call."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dest),
        ],
        desc=f"extract audio {dest.name}",
    )
    return dest


def detect_scene_changes(source: str | Path, threshold: float = 0.35) -> list[float]:
    """Timestamps (seconds) where ffmpeg's scene detector fires."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(source),
            "-filter:v", f"select='gt(scene,{threshold})',metadata=print:file=-",
            "-an", "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    times: list[float] = []
    for line in proc.stdout.splitlines():
        if "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except (IndexError, ValueError):
                continue
    return times
