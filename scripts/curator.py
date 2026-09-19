"""Optional LLM curation stage.

Sits between discovery and fetch. It takes the ranked candidates, enriches them
with lightweight metadata (description, top comments, optionally a transcript
snippet), and asks an OpenAI-compatible chat endpoint to:

  1. pick the best N clips for the compilation theme,
  2. explain why each was chosen,
  3. suggest a start timestamp for the highlight window.

The endpoint is OpenAI-compatible so it works with:

  - Ollama:        http://localhost:11434/v1
  - LocalAI
  - OpenAI
  - Any other /v1/chat/completions provider

If the LLM is unreachable, the stage is disabled, or parsing fails, the stage
falls back to the original discovery ranking unchanged.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any

from . import llm_client
from .config import Config, load_config
from .discover import Candidate, load_candidates, normalise_license
from .util import StageError, get_logger, read_json, write_json

log = get_logger("curator")

DEFAULT_OPENAI_BASE = "http://localhost:11434/v1"
DEFAULT_MODEL = "deepseek-v4-flash:cloud"


@dataclass
class CuratedCandidate(Candidate):
    """Candidate augmented with LLM curation output."""

    llm_reason: str = ""
    llm_suggested_start: float = 0.0
    llm_confidence: int = 0  # 1-10


# ---------------------------------------------------------------------------
# Metadata enrichment (cheap signals before we pay for an LLM call)
# ---------------------------------------------------------------------------

def _fetch_ytdlp_info(url: str, cfg: Config) -> dict[str, Any]:
    """Pull description, tags, channel and auto-captions without downloading."""
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("yt-dlp is required for curator enrichment") from exc

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "js_runtimes": {"node": {}},
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en"],
        "subtitlesformat": "json3",
    }
    if cfg.download.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.download.cookies_from_browser,)

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            return ydl.extract_info(url, download=False, process=False) or {}
        except Exception as exc:
            log.debug("yt-dlp enrichment failed for %s: %s", url, exc)
            return {}


def _first_english_subtitle_json(info: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first English auto/regular subtitle track we can find."""
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for source in (subs, auto):
        for lang in ("en", "en-US", "en-GB"):
            if lang in source and source[lang]:
                return source[lang][0]
    return None


def _transcript_from_json3_url(url: str, timeout: int = 15) -> str:
    """Fetch a json3 subtitle track and turn it into plain text."""
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        return ""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.debug("Could not fetch subtitle json3: %s", exc)
        return ""

    events = data.get("events") or []
    parts: list[str] = []
    for ev in events:
        for seg in ev.get("segs") or []:
            txt = seg.get("utf8")
            if txt:
                parts.append(txt)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _snippets(candidates: list[Candidate], cfg: Config) -> list[dict[str, Any]]:
    """Build a compact, LLM-readable summary for every candidate."""
    out: list[dict[str, Any]] = []
    for c in candidates:
        info = _fetch_ytdlp_info(c.url, cfg)
        description = (info.get("description") or "")[:500]
        transcript = ""
        sub = _first_english_subtitle_json(info)
        if sub:
            transcript = _transcript_from_json3_url(sub.get("url", ""))[:800]

        out.append(
            {
                "id": c.video_id,
                "title": c.title,
                "channel": c.channel,
                "views": c.views,
                "likes": c.likes,
                "comments": c.comments,
                "duration_sec": round(c.duration_sec, 1),
                "published_at": c.published_at,
                "virality_score": c.score,
                "license": c.license,
                "url": c.url,
                "description": description,
                "transcript_snippet": transcript,
            }
        )
    return out


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _parse_selection(raw: str, candidates: list[Candidate]) -> list[CuratedCandidate]:
    """Parse the LLM's JSON output and map it back to Candidate objects.

    Expected format (strict, inside a ```json block or bare):

        {
          "selected": [
            {
              "id": "abc123",
              "reason": "...",
              "suggested_start_sec": 12.5,
              "confidence": 8
            }
          ]
        }
    """
    by_id = {c.video_id: c for c in candidates}
    selected: list[CuratedCandidate] = []

    # Try to extract a JSON object from markdown fences first.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        # Otherwise take the first {...} block.
        match = re.search(r"(\{.*\})", raw, re.DOTALL)
        if match:
            raw = match.group(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("LLM output was not valid JSON: %s", exc)
        return selected

    for item in data.get("selected") or []:
        vid = item.get("id")
        original = by_id.get(vid)
        if not original:
            log.warning("LLM selected unknown video id %r", vid)
            continue
        cc = CuratedCandidate(**asdict(original))
        cc.llm_reason = str(item.get("reason", "")).strip()
        cc.llm_suggested_start = max(0.0, float(item.get("suggested_start_sec", 0.0)))
        cc.llm_confidence = max(1, min(10, int(item.get("confidence", 5))))
        selected.append(cc)

    return selected


def _build_prompt(
    snippets: list[dict[str, Any]], max_results: int, clip_seconds: float
) -> str:
    data = json.dumps(snippets, indent=2, ensure_ascii=False)
    return textwrap.dedent(
        f"""\
        You are a video compiler curator. Your job is to pick the best {max_results} clips
        for a viral "caught on camera / satisfying / funny moments" compilation.

        For each candidate you get: title, channel, view count, virality score, duration,
        a short description, and the first ~30 seconds of transcript (if available).

        Rules:
        - Prefer clips that feel surprising, funny, satisfying, or visually impressive.
        - Avoid repetition: if two clips seem like the same compilation channel reusing
          the same footage, pick the better one.
        - Respect licensing: prefer candidates with license "creativeCommon" when they
          are roughly as good as standard ones.
        - Suggest a highlight start time in seconds. It should land *inside* the video
          (less than duration - {clip_seconds}). Skip the very first 2 seconds unless
          the action starts immediately.

        Return ONLY a JSON object in this exact shape, no extra commentary:

        {{
          "selected": [
            {{
              "id": "VIDEO_ID",
              "reason": "one sentence why this clip was chosen",
              "suggested_start_sec": 12.5,
              "confidence": 8
            }}
          ]
        }}

        confidence is an integer 1-10.

        Candidates:
        {data}
        """
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def curate(
    cfg: Config | None = None, candidates: list[Candidate] | None = None
) -> list[CuratedCandidate]:
    cfg = cfg or load_config()
    cfg.make_dirs()

    if not getattr(cfg, "curator", None) or not cfg.curator.enabled:
        log.info("Curator disabled; passing discovery ranking through")
        candidates = candidates or load_candidates(cfg)
        return [CuratedCandidate(**asdict(c)) for c in candidates]

    candidates = candidates or load_candidates(cfg)
    if not candidates:
        raise StageError("No candidates to curate - run discover first")

    c = cfg.curator
    max_review = min(c.max_candidates, len(candidates))
    review_pool = candidates[:max_review]
    log.info(
        "Curating %d of %d candidates with %s/%s",
        len(review_pool),
        len(candidates),
        c.base_url,
        c.model,
    )

    snippets = _snippets(review_pool, cfg)
    prompt = _build_prompt(snippets, cfg.discovery.max_results, cfg.highlight.clip_seconds)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful video curation assistant. You always return valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]

    completion = llm_client.chat_completion(messages, cfg)
    raw = llm_client.extract_message_text(completion)
    if not raw.strip():
        log.warning("LLM returned empty content and reasoning")
        return [CuratedCandidate(**asdict(c)) for c in candidates]

    selected = _parse_selection(raw, review_pool)
    if not selected:
        log.warning("LLM returned no usable selection; falling back to discovery ranking")
        return [CuratedCandidate(**asdict(c)) for c in candidates]

    # If the LLM returned fewer than max_results, top up from the discovery ranking
    # so the pipeline always has a predictable number of clips to render.
    chosen_ids = {s.video_id for s in selected}
    for c in candidates:
        if c.video_id in chosen_ids:
            continue
        if len(selected) >= cfg.discovery.max_results:
            break
        selected.append(CuratedCandidate(**asdict(c)))

    for i, s in enumerate(selected, 1):
        log.info(
            "#%d  %-6.1f  %-45.45s  [LLM confidence %d/10] %s",
            i,
            s.score,
            s.title,
            s.llm_confidence,
            s.llm_reason,
        )

    out = cfg.path("workspace") / "curated_candidates.json"
    write_json(out, [asdict(s) for s in selected])
    log.info("Wrote %d curated candidates to %s", len(selected), out)
    return selected


def load_curated_candidates(cfg: Config) -> list[CuratedCandidate]:
    path = cfg.path("workspace") / "curated_candidates.json"
    data = read_json(path)
    if not data:
        raise StageError(f"{path} not found or empty - run the curator stage first")
    return [CuratedCandidate(**item) for item in data]


if __name__ == "__main__":
    curate()
