"""Stage 2 - download the selected candidates with the yt-dlp CLI.

Every download records an ``Asset`` carrying the attribution metadata (uploader,
channel URL, licence) that the credits card and description are built from, so a
finished compilation can always name its sources.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import llm_client
from .config import Config, load_config
from .curator import CuratedCandidate, load_curated_candidates
from .discover import Candidate, load_candidates, normalise_license
from .util import StageError, get_logger, read_json, slugify, write_json

log = get_logger("fetch")


@dataclass
class Asset:
    video_id: str
    path: str
    title: str
    channel: str
    channel_url: str
    source_url: str
    license: str
    duration_sec: float
    platform: str = "youtube"
    score: float = 0.0
    llm_suggested_start: float = 0.0


def _find_ytdlp() -> str:
    """Locate the yt-dlp binary; prefer the active venv if present."""
    # Common names / locations
    for candidate in ("yt-dlp", "youtube-dl"):
        path = shutil.which(candidate)
        if path:
            return path
    raise StageError(
        "yt-dlp binary not found on PATH. Install it (pip install yt-dlp) and retry."
    )


YTDLP = _find_ytdlp()


@dataclass
class _Command:
    exe: str
    args: list[str]
    env: dict[str, str] | None = None


def _run(cmd: _Command, desc: str, timeout: int | None = 300) -> subprocess.CompletedProcess[str]:
    """Run the yt-dlp CLI, raising StageError with stderr on failure."""
    full = [cmd.exe, *cmd.args]
    log.debug("$ %s", " ".join(full))
    try:
        proc = subprocess.run(
            full,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=cmd.env,
        )
    except subprocess.TimeoutExpired as exc:
        raise StageError(f"{desc}: timed out after {timeout}s") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-20:])
        raise StageError(f"{desc} failed (exit {proc.returncode}):\n{tail}")
    return proc


def _base_args(cfg: Config) -> list[str]:
    """Common yt-dlp flags derived from config."""
    args = [
        "--no-playlist",
        "--retries", str(cfg.download.retries),
        "--fragment-retries", str(cfg.download.retries),
        "--extractor-args", "youtube:player_client=android",
        "--js-runtimes", "node",
        "--newline",
        "--no-warnings",
    ]
    if cfg.download.max_filesize_mb:
        args += ["--max-filesize", f"{cfg.download.max_filesize_mb}M"]
    if cfg.download.rate_limit:
        args += ["--limit-rate", cfg.download.rate_limit]
    if cfg.download.cookies_from_browser:
        args += ["--cookies-from-browser", cfg.download.cookies_from_browser]
    return args


def _ydl_command(cfg: Config, outtmpl: str, url: str, *, fmt: str | None = None) -> _Command:
    """Build a download command. fmt=None uses cfg.download.format."""
    args = _base_args(cfg)
    args += [
        "--format", fmt or cfg.download.format,
        "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "--output", outtmpl,
        url,
    ]
    return _Command(YTDLP, args)


def _parse_list_formats(stderr: str, stdout: str) -> list[dict[str, Any]]:
    """Parse the ASCII table printed by --list-formats into a compact list."""
    # yt-dlp prints the table to stderr in quiet mode, or stdout otherwise.
    text = stderr + "\n" + stdout
    formats: list[dict[str, Any]] = []
    in_table = False
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("[info] Available formats"):
            in_table = True
            continue
        if in_table and (line.startswith("─") or line.startswith("-")):
            continue
        if in_table and line.startswith("ID "):
            continue
        if in_table:
            # Heuristic parse of the fixed-width-ish table.
            parts = line.split("│")
            if len(parts) < 4:
                # Maybe the line is space-separated fallback.
                parts = line.split()
                if len(parts) >= 4 and parts[0] not in ("sb0", "sb1", "sb2", "sb3"):
                    formats.append({
                        "id": parts[0],
                        "ext": parts[1] if len(parts) > 1 else "",
                        "resolution": parts[2] if len(parts) > 2 else "",
                        "rest": " ".join(parts[3:]),
                    })
                continue
            left = [p.strip() for p in parts[0].split()]
            if not left or left[0] in ("sb0", "sb1", "sb2", "sb3") or left[0].startswith("sb"):
                continue
            fmt_id = left[0]
            ext = left[1] if len(left) > 1 else ""
            resolution = left[2] if len(left) > 2 else ""
            rest = "│".join(parts[1:]).strip()
            formats.append({"id": fmt_id, "ext": ext, "resolution": resolution, "rest": rest})
    return formats


def _existing_download(dest_dir: Path, video_id: str) -> Path | None:
    for path in dest_dir.glob(f"{video_id}__*"):
        if path.suffix.lower() in {".mp4", ".mkv", ".webm"} and path.stat().st_size > 0:
            return path
    return None


def _candidate_start(candidate: Candidate) -> float:
    """Preserve an LLM-suggested start timestamp if the candidate carries one."""
    if isinstance(candidate, CuratedCandidate):
        return candidate.llm_suggested_start
    return 0.0


def _llm_pick_format(formats: list[dict[str, Any]], cfg: Config) -> str | None:
    """Ask the LLM to choose a yt-dlp format id (or id+id pair) for editing."""
    if not cfg.download.llm_format_fallback:
        return None
    if not formats:
        return None

    # Prefer not to overwhelm the LLM; drop storyboards/rows without an id.
    usable = [f for f in formats if f.get("id") and not str(f.get("id")).startswith("sb")]
    if not usable:
        return None
    max_results = min(len(usable), 15)
    data = json.dumps(usable[:max_results], indent=2, ensure_ascii=False)
    prompt = textwrap.dedent(
        f"""\
        You are choosing a YouTube download format for video editing and subtitling.

        Pick the best format id or id+id pair from the list below. Preferences, in order:
        1. A single combined MP4 stream with H.264 video and AAC audio.
        2. A format that contains both video and audio in one stream.
        3. A pair "video_id+audio_id" where video is H.264/MP4 and audio is AAC/M4A.
        Avoid av1, vp9-only, or audio-only formats unless nothing else exists.

        Return ONLY a JSON object in this exact shape, no commentary:

        {{
          "format": "..."
        }}

        Available formats:
        {data}
        """
    )
    messages = [
        {"role": "system", "content": "You are a JSON generator. You always return exactly one JSON object, nothing else."},
        {"role": "user", "content": prompt},
    ]
    completion = llm_client.chat_completion(messages, cfg, temperature=0.1, max_tokens=256)
    raw = llm_client.extract_message_text(completion)
    data = llm_client.parse_json_object(raw) if raw else None
    picked = data.get("format") if isinstance(data, dict) else None
    if not isinstance(picked, str) or not picked.strip():
        log.warning("LLM did not return a usable format")
        return None
    picked = picked.strip()
    # Strip any explanatory prefix the model may have added ("format 18",
    # "id: 18", etc.) and keep only the actual format id.
    m = re.search(r"([a-zA-Z0-9_+-]+)", picked)
    if m:
        picked = m.group(1)
    log.info("LLM picked format %r", picked)
    return picked


def _extract_info_json(url: str, cfg: Config, timeout: int = 60) -> dict[str, Any] | None:
    """Dump metadata as JSON via the yt-dlp CLI."""
    args = _base_args(cfg) + ["--dump-json", "--skip-download", url]
    cmd = _Command(YTDLP, args)
    try:
        proc = _run(cmd, f"metadata {url}", timeout=timeout)
    except StageError as exc:
        log.warning("Could not fetch metadata: %s", exc)
        return None
    # --dump-json emits one JSON object per line; take the last non-empty line.
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _try_download(candidate: Candidate, cfg: Config, outtmpl: str) -> tuple[Path, dict[str, Any]] | None:
    """Attempt a CLI download, optionally retrying with an LLM-chosen format."""
    cmd = _ydl_command(cfg, outtmpl, candidate.url)
    log.info("Fetching %s - %.50s", candidate.video_id, candidate.title)
    try:
        _run(cmd, f"download {candidate.url}")
    except StageError as exc:
        err = str(exc)
        if (
            "Requested format is not available" in err
            or "Only images are available for download" in err
        ) and cfg.download.llm_format_fallback:
            log.warning("Default format unavailable for %s; listing formats", candidate.video_id)
            list_args = _base_args(cfg) + ["--list-formats", "--skip-download", candidate.url]
            try:
                proc = _run(_Command(YTDLP, list_args), f"list-formats {candidate.url}", timeout=120)
            except StageError as list_exc:
                log.warning("Could not list formats for %s: %s", candidate.video_id, list_exc)
                proc = None
            if proc:
                formats = _parse_list_formats(proc.stderr, proc.stdout)
                picked = _llm_pick_format(formats, cfg)
                if picked:
                    retry = _ydl_command(cfg, outtmpl, candidate.url, fmt=picked)
                    try:
                        _run(retry, f"download {candidate.url} (LLM format)")
                    except StageError as retry_exc:
                        log.warning("LLM-chosen format failed for %s: %s", candidate.video_id, retry_exc)
                        return None
                else:
                    return None
            else:
                return None
        else:
            log.error("Download failed for %s: %s", candidate.url, exc)
            return None

    info = _extract_info_json(candidate.url, cfg)
    if info is None:
        log.error("No metadata returned for %s", candidate.url)
        return None

    dest_dir = Path(outtmpl).parent
    downloaded = _existing_download(dest_dir, candidate.video_id)
    if downloaded is None:
        log.error("yt-dlp reported success but no file landed for %s", candidate.video_id)
        return None
    return downloaded, info


def _download_path_name(platform: str) -> str:
    """Map a candidate platform to the configured download directory name."""
    return {
        "twitch": "twitch_downloads",
        "music_video": "youtube_music_downloads",
        "instagram": "instagram_downloads",
    }.get(platform, "downloads")


def download_one(candidate: Candidate, cfg: Config) -> Asset | None:
    dest_dir = cfg.path(_download_path_name(candidate.platform))
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{candidate.video_id}__{slugify(candidate.title, 40)}"
    cached = _existing_download(dest_dir, candidate.video_id)
    if cached:
        log.info("Cached  %s", cached.name)
        info = read_json(dest_dir / f"{candidate.video_id}.info.json", default={}) or {}
        return Asset(
            video_id=candidate.video_id,
            path=str(cached),
            title=info.get("title") or candidate.title,
            channel=info.get("uploader") or candidate.channel,
            channel_url=info.get("uploader_url") or info.get("channel_url") or "",
            source_url=candidate.url,
            license=normalise_license(info.get("license")) if info.get("license") else candidate.license,
            duration_sec=float(info.get("duration") or candidate.duration_sec or 0.0),
            platform=candidate.platform,
            score=candidate.score,
            llm_suggested_start=_candidate_start(candidate),
        )

    outtmpl = str(dest_dir / f"{stem}.%(ext)s")
    result = _try_download(candidate, cfg, outtmpl)
    if result is None:
        return None
    downloaded, info = result

    # Keep the raw metadata next to the media; the credits stage reads it back.
    write_json(dest_dir / f"{candidate.video_id}.info.json", {
        k: info.get(k)
        for k in ("id", "title", "uploader", "uploader_url", "channel_url", "webpage_url",
                  "license", "duration", "view_count", "like_count", "upload_date", "tags")
    })

    return Asset(
        video_id=candidate.video_id,
        path=str(downloaded),
        title=info.get("title") or candidate.title,
        channel=info.get("uploader") or candidate.channel,
        channel_url=info.get("uploader_url") or info.get("channel_url") or "",
        source_url=info.get("webpage_url") or candidate.url,
        # yt-dlp reports a human string ('Creative Commons Attribution...'),
        # the API a slug; the credits stage matches on the slug.
        license=normalise_license(info.get("license")) if info.get("license") else candidate.license,
        duration_sec=float(info.get("duration") or candidate.duration_sec or 0.0),
        platform=candidate.platform,
        score=candidate.score,
        llm_suggested_start=_candidate_start(candidate),
    )


def fetch(cfg: Config | None = None, candidates: list[Candidate] | None = None) -> list[Asset]:
    cfg = cfg or load_config()
    cfg.make_dirs()
    candidates = candidates or load_candidates(cfg)

    assets: list[Asset] = []
    for candidate in candidates:
        asset = download_one(candidate, cfg)
        if asset:
            assets.append(asset)

    if not assets:
        raise StageError(
            "Every download failed. Check connectivity, credentials (YouTube/Twitch/Instagram), or set "
            "download.cookies_from_browser in config.yaml if a platform is asking for sign-in."
        )

    out = cfg.path("workspace") / "assets.json"
    write_json(out, [asdict(a) for a in assets])
    platforms = sorted({a.platform for a in assets})
    path_msgs = [f"{p}: {cfg.path(_download_path_name(p))}" for p in platforms]
    log.info("Downloaded %d/%d videos -> %s", len(assets), len(candidates), "; ".join(path_msgs))
    return assets


def load_assets(cfg: Config) -> list[Asset]:
    data = read_json(cfg.path("workspace") / "assets.json")
    if not data:
        raise StageError("assets.json not found or empty - run the fetch stage first")
    return [Asset(**item) for item in data]


if __name__ == "__main__":
    fetch()
