"""Stage 1 - find videos that are actually going viral right now.

The original implementation scraped YouTube's search HTML with hard-coded CSS
class names and asked pytube for a "trending channel". Both break constantly:
YouTube renders results client-side and has no channel at /feed/trending.

This version uses the YouTube Data API v3 (authoritative view/like/comment
counts in one request) with a keyless yt-dlp search fallback.
"""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import quote_plus

from . import llm_client
from .config import Config, load_config
from .util import StageError, get_logger, read_json, write_json

log = get_logger("discover")

API_BASE = "https://www.googleapis.com/youtube/v3"
ISO_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
)
# YouTube's own "Creative Commons" search filter (the `sp` blob the web UI sets
# when you tick Features > Creative Commons). Narrows the pool before we pay for
# per-video metadata lookups; the license is still verified per video below.
CC_SEARCH_URL = "https://www.youtube.com/results?search_query={query}&sp=EgIwAQ%3D%3D"


@dataclass
class Candidate:
    video_id: str
    url: str
    title: str
    channel: str
    channel_id: str = ""
    views: int = 0
    likes: int = 0
    comments: int = 0
    duration_sec: float = 0.0
    published_at: str = ""
    license: str = "youtube"
    platform: str = "youtube"  # "youtube" | "twitch" | ...
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    # True once the counts came from a per-video lookup rather than a flat
    # search result, so `views == 0` means "no views", not "not reported".
    metadata_complete: bool = False

    @property
    def age_hours(self) -> float:
        if not self.published_at:
            return 24.0 * 30
        try:
            published = datetime.fromisoformat(self.published_at.replace("Z", "+00:00"))
        except ValueError:
            return 24.0 * 30
        delta = datetime.now(timezone.utc) - published
        return max(delta.total_seconds() / 3600.0, 0.5)


def parse_iso_duration(value: str | None) -> float:
    """'PT1M30S' -> 90.0 seconds."""
    if not value:
        return 0.0
    match = ISO_DURATION.match(value)
    if not match:
        return 0.0
    parts = {k: int(v) for k, v in match.groupdict(default="0").items()}
    return float(
        parts["days"] * 86400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]
    )


def virality_score(c: Candidate) -> tuple[float, list[str]]:
    """Rank by how fast a video is accumulating attention, not by raw view count.

    A 3-year-old video with 50M views is popular; a 12-hour-old video with 2M
    views is viral. The score blends three signals:

      velocity   views per hour since upload (log-scaled, dominant term)
      engagement (likes + 3x comments) / views - how strongly people react
      freshness  decay that favours the last few days
    """
    velocity = c.views / c.age_hours
    engagement = (c.likes + 3 * c.comments) / max(c.views, 1)
    freshness = math.exp(-c.age_hours / 168.0)  # half-life of roughly a week

    score = (
        math.log10(velocity + 1) * 10.0
        + min(engagement, 0.2) * 100.0
        + freshness * 15.0
    )
    reasons = [
        f"{c.views:,} views",
        f"{velocity:,.0f} views/hr",
        f"{engagement * 100:.1f}% engagement",
        f"{c.age_hours:.0f}h old",
    ]
    return round(score, 3), reasons


def _llm_generate_search_queries(cfg: Config) -> list[str]:
    """Ask the LLM for a fresh, diverse set of YouTube search queries."""
    d = cfg.discovery
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = textwrap.dedent(
        f"""\
        You are a YouTube query generator. Return ONLY a JSON object — no explanation,
        no markdown, no chain of thought. The response must start with '{{' and end with '}}'.

        Task: generate {d.llm_query_count} short, realistic YouTube search queries
        that will surface recent, high-engagement viral clips for a compilation about
        "{d.llm_query_theme}".

        Context: region {d.region}, today {today}.
        Rules:
        - Each query is a phrase a real person would type into YouTube search.
        - Mix genres: funny fails, satisfying moments, unexpected events, impressive
          skills, caught-on-camera, wholesome, weird, dramatic, etc.
        - Vary phrasing so the result set changes run-to-run.

        Required JSON shape:

        {{
          "queries": ["query one", "query two", "..."]
        }}
        """
    )
    messages = [
        {
            "role": "system",
            "content": "You are a JSON generator. You always return exactly one JSON object, nothing else.",
        },
        {"role": "user", "content": prompt},
    ]

    completion = llm_client.chat_completion(
        messages, cfg, temperature=d.llm_query_temperature, max_tokens=1024
    )
    raw = llm_client.extract_message_text(completion)
    data = llm_client.parse_json_object(raw) if raw else None
    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list) or not queries:
        log.warning("LLM did not return usable queries; falling back to static list")
        return []

    cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    log.info("LLM generated %d search queries for theme %r", len(cleaned), d.llm_query_theme)
    for i, q in enumerate(cleaned, 1):
        log.info("  %2d. %s", i, q)
    return cleaned[: d.llm_query_count]


def _search_queries_for_run(cfg: Config) -> list[str]:
    """Return LLM-generated queries when enabled, otherwise the static config list."""
    d = cfg.discovery
    if d.llm_query_enabled:
        generated = _llm_generate_search_queries(cfg)
        if generated:
            return generated
        log.warning("LLM query generation failed; using static search_queries")
    return d.search_queries


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def normalise_license(raw: str | None) -> str:
    """yt-dlp reports a human string; the API reports a slug. Speak one language.

    yt-dlp: 'Creative Commons Attribution license (reuse allowed)' or None
    API:    'creativeCommon' or 'youtube'
    """
    if not raw:
        return "youtube"
    lowered = raw.lower()
    if "creative commons" in lowered or lowered == "creativecommon":
        return "creativeCommon"
    return "youtube"


def _hydrate_metadata(
    candidates: list[Candidate], cfg: Config, *, trust_metadata: bool = True
) -> list[Candidate]:
    """Fill in the fields a flat search result does not carry.

    yt-dlp's ``extract_flat`` search gives id/title/channel/views/duration and
    nothing else - crucially no ``license``, so every candidate keeps the
    default 'youtube' and a creativeCommon filter drops the entire pool. One
    metadata call per video (``process=False``, so no format resolution) fills
    the license plus the upload date and engagement counts the ranking wants.
    """
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("yt-dlp is required (pip install -r requirements.txt)") from exc

    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,  # age-gated / removed videos must not kill the batch
        "extractor_args": {"youtube": {"player_client": ["android"]}},  # avoid string -> char iteration
        "js_runtimes": {"node": {}},
    }
    if cfg.download.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.download.cookies_from_browser,)

    log.info("Fetching licenses for %d candidates (~1s each)", len(candidates))
    hydrated: list[Candidate] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for c in candidates:
            try:
                info = ydl.extract_info(c.url, download=False, process=False)
            except Exception as exc:  # yt-dlp raises a wide variety of errors
                log.debug("Metadata lookup failed for %s: %s", c.video_id, exc)
                info = None
            if not info:
                # Unknown license: keep the default 'youtube' so a CC filter
                # rejects it and the drop shows up in the filter summary.
                hydrated.append(c)
                continue
            c.license = normalise_license(info.get("license"))
            c.views = int(info.get("view_count") or c.views)
            c.likes = int(info.get("like_count") or 0)
            c.comments = int(info.get("comment_count") or 0)
            c.duration_sec = float(info.get("duration") or c.duration_sec)
            c.title = info.get("title") or c.title
            c.channel = info.get("channel") or info.get("uploader") or c.channel
            c.channel_id = info.get("channel_id") or c.channel_id
            c.metadata_complete = trust_metadata
            upload_date = info.get("upload_date")
            if upload_date and len(upload_date) == 8:
                c.published_at = (
                    f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:]}T00:00:00Z"
                )
            hydrated.append(c)
    return hydrated


def _api_get(endpoint: str, params: dict[str, Any], api_key: str) -> dict:
    try:
        import requests  # noqa: PLC0415 - only needed for the API source
    except ImportError as exc:
        raise StageError("`requests` is required for the youtube_api source (pip install requests)") from exc

    params = {**params, "key": api_key}
    response = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=30)
    if response.status_code == 403:
        raise StageError(
            "YouTube API rejected the request (403). Usual causes: quota exhausted, "
            "the API key is restricted, or the Data API v3 is not enabled for the project."
        )
    response.raise_for_status()
    return response.json()


def _candidates_from_api_items(items: Iterable[dict]) -> list[Candidate]:
    out: list[Candidate] = []
    for item in items:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})
        status = item.get("status", {})
        video_id = item.get("id")
        if isinstance(video_id, dict):  # search.list shape
            video_id = video_id.get("videoId")
        if not video_id:
            continue
        out.append(
            Candidate(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=snippet.get("title", "Untitled"),
                channel=snippet.get("channelTitle", "Unknown"),
                channel_id=snippet.get("channelId", ""),
                views=int(stats.get("viewCount", 0) or 0),
                likes=int(stats.get("likeCount", 0) or 0),
                comments=int(stats.get("commentCount", 0) or 0),
                duration_sec=parse_iso_duration(content.get("duration")),
                published_at=snippet.get("publishedAt", ""),
                license=normalise_license(status.get("license")),
            )
        )
    return out


def from_youtube_api(cfg: Config) -> list[Candidate]:
    d = cfg.discovery
    if not cfg.youtube_api_key:
        raise StageError(
            "discovery.source is 'youtube_api' but YOUTUBE_API_KEY is not set. "
            "Add it to .env, or set discovery.source: ytdlp_search in config.yaml."
        )

    candidates: list[Candidate] = []

    # 1) The regional "most popular" chart - the closest thing to an official
    #    trending feed, and it comes back with full statistics attached.
    chart_params: dict[str, Any] = {
        "part": "snippet,statistics,contentDetails,status",
        "chart": "mostPopular",
        "regionCode": d.region,
        "maxResults": min(d.max_candidates, 50),
    }
    if d.category_id:
        chart_params["videoCategoryId"] = str(d.category_id)
    candidates += _candidates_from_api_items(_api_get("videos", chart_params, cfg.youtube_api_key).get("items", []))

    # 2) Recent high-view results per query, which surfaces viral clips that
    #    never make the national chart.
    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=d.published_within_hours)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    queries = _search_queries_for_run(cfg)
    for query in queries:
        search_params: dict[str, Any] = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "viewCount",
            "publishedAfter": published_after,
            "maxResults": 25,
            "regionCode": d.region,
        }
        if d.license_filter == "creativeCommon":
            search_params["videoLicense"] = "creativeCommon"
        try:
            found = _api_get("search", search_params, cfg.youtube_api_key).get("items", [])
        except StageError as exc:
            log.warning("Search for %r failed: %s", query, exc)
            continue

        ids = [i["id"]["videoId"] for i in found if i.get("id", {}).get("videoId")]
        if not ids:
            continue
        # search.list returns no statistics; hydrate in one batched videos.list call.
        hydrated = _api_get(
            "videos",
            {"part": "snippet,statistics,contentDetails,status", "id": ",".join(ids[:50])},
            cfg.youtube_api_key,
        ).get("items", [])
        candidates += _candidates_from_api_items(hydrated)
        log.info("Query %-28r -> %d candidates", query, len(hydrated))

    return candidates


def from_ytdlp_search(cfg: Config) -> list[Candidate]:
    """Keyless fallback. Fewer signals (no like/comment counts) but zero setup."""
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("yt-dlp is required (pip install -r requirements.txt)") from exc

    d = cfg.discovery
    want_cc = d.license_filter == "creativeCommon"
    queries = _search_queries_for_run(cfg)
    per_query = max(1, d.max_candidates // max(len(queries), 1))
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True}
    if cfg.download.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.download.cookies_from_browser,)

    candidates: list[Candidate] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for query in queries:
            if want_cc:
                # ytsearch: has no license filter, so go through the results page
                # with YouTube's own Creative Commons filter applied.
                target = CC_SEARCH_URL.format(query=quote_plus(query))
            else:
                target = f"ytsearch{per_query}:{query}"
            try:
                result = ydl.extract_info(target, download=False)
            except Exception as exc:  # yt-dlp raises a wide variety of errors
                log.warning("Search %r failed: %s", query, exc)
                continue
            found = 0
            for entry in result.get("entries") or []:
                if not entry or not entry.get("id"):
                    continue
                candidates.append(
                    Candidate(
                        video_id=entry["id"],
                        url=entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}",
                        title=entry.get("title") or "Untitled",
                        channel=entry.get("channel") or entry.get("uploader") or "Unknown",
                        channel_id=entry.get("channel_id") or "",
                        views=int(entry.get("view_count") or 0),
                        duration_sec=float(entry.get("duration") or 0.0),
                        # Flat search results carry no upload date; assume "recent"
                        # so freshness does not distort the ranking either way.
                        published_at="",
                    )
                )
                found += 1
                if want_cc and found >= per_query:
                    break
            log.info("Query %-28r -> %d candidates", query, found)

    if want_cc:
        # Cheap filters first, so we only pay for metadata on plausible clips,
        # and only once per video no matter how many queries surfaced it.
        log.info("Collected %d raw candidates; checking licenses", len(candidates))
        shortlist = apply_filters(_dedupe(candidates), cfg, check_license=False)
        candidates = _hydrate_metadata(shortlist[: d.max_candidates], cfg)
    return candidates


def _music_video_queries(cfg: Config) -> list[str]:
    """Build YouTube search queries that always return music + video clips."""
    m = cfg.music_video
    queries: list[str] = []
    for base in m.queries:
        # Force a visual/video term so results are never audio-only tracks.
        queries.append(base)
        if m.use_shorts:
            queries.append(f"{base} shorts")
    for genre in m.genres:
        queries.append(f"viral {genre} music video")
        if m.use_shorts:
            queries.append(f"viral {genre} music video shorts")
    # De-duplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out[: max(len(m.queries) * 2, 12)]


def from_music_video(cfg: Config) -> list[Candidate]:
    """Discover viral music VIDEO clips on YouTube.

    Uses the YouTube Data API when YOUTUBE_API_KEY is set, targeting the
    Music category (ID 10) and music-video-specific queries. Falls back to
    yt-dlp search when the API is unavailable. Every candidate is a video,
    never an audio-only track, and carries ``platform="music_video"`` so the
    fetch stage lands it in the music-video download folder.
    """
    queries = _music_video_queries(cfg)
    log.info("Music video queries: %s", queries)

    if cfg.youtube_api_key:
        candidates: list[Candidate] = []
        # 1) Regional Music chart (these are music videos, not audio tracks).
        chart_params: dict[str, Any] = {
            "part": "snippet,statistics,contentDetails,status",
            "chart": "mostPopular",
            "regionCode": cfg.discovery.region,
            "videoCategoryId": "10",
            "maxResults": min(cfg.discovery.max_candidates, 50),
        }
        try:
            chart_items = _api_get("videos", chart_params, cfg.youtube_api_key).get("items", [])
            candidates += _candidates_from_api_items(chart_items)
            log.info("YouTube Music chart -> %d candidates", len(chart_items))
        except StageError as exc:
            log.warning("YouTube Music chart failed: %s", exc)

        # 2) Search per music-video query.
        published_after = (
            datetime.now(timezone.utc) - timedelta(hours=cfg.discovery.published_within_hours)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        per_query = max(1, cfg.discovery.max_candidates // max(len(queries), 1))
        for query in queries:
            search_params: dict[str, Any] = {
                "part": "snippet",
                "q": query,
                "type": "video",
                "order": "viewCount",
                "publishedAfter": published_after,
                "maxResults": min(per_query, 25),
                "regionCode": cfg.discovery.region,
            }
            if cfg.discovery.license_filter == "creativeCommon":
                search_params["videoLicense"] = "creativeCommon"
            try:
                found = _api_get("search", search_params, cfg.youtube_api_key).get("items", [])
            except StageError as exc:
                log.warning("Search %r failed: %s", query, exc)
                continue
            ids = [i["id"]["videoId"] for i in found if i.get("id", {}).get("videoId")]
            if not ids:
                continue
            hydrated = _api_get(
                "videos",
                {"part": "snippet,statistics,contentDetails,status", "id": ",".join(ids[:50])},
                cfg.youtube_api_key,
            ).get("items", [])
            candidates += _candidates_from_api_items(hydrated)
            log.info("Music video query %-28r -> %d candidates", query, len(hydrated))

        for c in candidates:
            c.platform = "music_video"
        return candidates

    # API key absent: fall back to yt-dlp search with music-video queries.
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("yt-dlp is required (pip install -r requirements.txt)") from exc

    per_query = max(1, cfg.discovery.max_candidates // max(len(queries), 1))
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True}
    if cfg.download.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.download.cookies_from_browser,)

    candidates: list[Candidate] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for query in queries:
            target = f"ytsearch{per_query}:{query}"
            try:
                result = ydl.extract_info(target, download=False)
            except Exception as exc:
                log.warning("Music video search %r failed: %s", query, exc)
                continue
            for entry in result.get("entries") or []:
                if not entry or not entry.get("id"):
                    continue
                candidates.append(
                    Candidate(
                        video_id=entry["id"],
                        url=entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}",
                        title=entry.get("title") or "Untitled",
                        channel=entry.get("channel") or entry.get("uploader") or "Unknown",
                        channel_id=entry.get("channel_id") or "",
                        views=int(entry.get("view_count") or 0),
                        duration_sec=float(entry.get("duration") or 0.0),
                        published_at="",
                        license="youtube" if cfg.discovery.license_filter == "any" else "",
                        platform="music_video",
                    )
                )
            log.info("Music video query %-28r -> %d candidates", query, len(result.get("entries") or []))

    if cfg.discovery.license_filter == "creativeCommon":
        log.info("Collected %d raw music video candidates; checking licenses", len(candidates))
        shortlist = apply_filters(_dedupe(candidates), cfg, check_license=False)
        candidates = _hydrate_metadata(shortlist[: cfg.discovery.max_candidates], cfg)
        for c in candidates:
            c.platform = "music_video"
    return candidates


def from_file(cfg: Config) -> list[Candidate]:
    """Hand-picked URLs, one per line. The safest source: you chose every video."""
    path = cfg.root / cfg.discovery.sources_file
    if not path.exists():
        raise StageError(f"discovery.source is 'file' but {path} does not exist")

    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        match = re.search(r"(?:v=|youtu\.be/|shorts/|/embed/)([\w-]{11})", url)
        video_id = match.group(1) if match else url
        candidates.append(
            Candidate(video_id=video_id, url=url, title=f"manual:{video_id}", channel="manual")
        )

    # A hand-picked list carries no metadata at all, so a license filter would
    # reject every line of it. Look the videos up first. The clips stay exempt
    # from the popularity floor - you already chose them - so the looked-up
    # counts inform ranking but never filtering.
    if cfg.discovery.license_filter == "creativeCommon":
        candidates = _hydrate_metadata(candidates, cfg, trust_metadata=False)
    return candidates


# ---------------------------------------------------------------------------
# Twitch sources
# ---------------------------------------------------------------------------

TWITCH_AUTH_URL = "https://id.twitch.tv/oauth2/token"
TWITCH_API_BASE = "https://api.twitch.tv/helix"


def _twitch_token(cfg: Config) -> str:
    """Fetch an app-access token for the configured Twitch credentials."""
    if not cfg.twitch_client_id or not cfg.twitch_client_secret:
        raise StageError(
            "discovery.source is 'twitch' but TWITCH_CLIENT_ID and/or TWITCH_CLIENT_SECRET "
            "are not set. Add them to .env and retry."
        )
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("`requests` is required for the twitch source (pip install requests)") from exc

    resp = requests.post(
        TWITCH_AUTH_URL,
        params={
            "client_id": cfg.twitch_client_id,
            "client_secret": cfg.twitch_client_secret,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    if resp.status_code in (400, 401, 403):
        raise StageError(
            f"Twitch API rejected the credentials (HTTP {resp.status_code}). "
            "Check TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET."
        )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise StageError("Twitch token response contained no access_token")
    return str(token)


def _twitch_headers(cfg: Config, token: str) -> dict[str, str]:
    return {
        "Client-ID": cfg.twitch_client_id or "",
        "Authorization": f"Bearer {token}",
    }


def _twitch_get(endpoint: str, cfg: Config, token: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("`requests` is required for the twitch source (pip install requests)") from exc

    url = f"{TWITCH_API_BASE}/{endpoint}"
    resp = requests.get(url, headers=_twitch_headers(cfg, token), params=params, timeout=30)
    if resp.status_code == 401:
        raise StageError(
            "Twitch API returned 401. Credentials may be invalid or the token expired."
        )
    resp.raise_for_status()
    return resp.json()


def _twitch_period_range(period: str) -> tuple[str | None, str | None]:
    """Map a human period to ISO UTC bounds for the Helix /clips endpoint."""
    now = datetime.now(timezone.utc)
    if period == "day":
        delta = timedelta(days=1)
    elif period == "week":
        delta = timedelta(days=7)
    elif period == "month":
        delta = timedelta(days=30)
    else:
        return None, None
    return (now - delta).isoformat(), now.isoformat()


def _twitch_game_id(game_name: str, cfg: Config, token: str) -> str | None:
    data = _twitch_get("games", cfg, token, {"name": game_name})
    items = data.get("data") or []
    if not items:
        log.warning("Twitch game %r not found; skipping", game_name)
        return None
    return str(items[0]["id"])


def _twitch_top_games(cfg: Config, token: str, first: int = 10) -> list[dict[str, Any]]:
    data = _twitch_get("games/top", cfg, token, {"first": first})
    return list(data.get("data") or [])


def _twitch_clips(
    cfg: Config,
    token: str,
    game_id: str,
    started_at: str | None,
    ended_at: str | None,
    language: str | None,
    max_results: int,
) -> list[dict[str, Any]]:
    """Paginate through /clips for a single game, returning up to max_results."""
    params: dict[str, Any] = {"game_id": game_id, "first": min(max_results, 100)}
    if started_at:
        params["started_at"] = started_at
    if ended_at:
        params["ended_at"] = ended_at
    if language:
        params["language"] = language

    clips: list[dict[str, Any]] = []
    after: str | None = None
    while len(clips) < max_results:
        if after:
            params["after"] = after
        elif "after" in params:
            del params["after"]
        data = _twitch_get("clips", cfg, token, params)
        batch = data.get("data") or []
        if not batch:
            break
        clips.extend(batch)
        after = (data.get("pagination") or {}).get("cursor")
        if not after:
            break
    return clips[:max_results]


def from_twitch_api(cfg: Config) -> list[Candidate]:
    """Discover trending Twitch clips via the Helix API.

    Requires TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET. If ``twitch.game_name``
    is set, clips are fetched for that game only; otherwise the top games on
    Twitch are queried in parallel (in the request sense) and their top clips
    are merged. The existing ``discovery.min_views`` and duration filters are
    reused, so the same popularity floor applies.
    """
    token = _twitch_token(cfg)
    started_at, ended_at = _twitch_period_range(cfg.twitch.period)

    if cfg.twitch.game_name:
        game_id = _twitch_game_id(cfg.twitch.game_name, cfg, token)
        if not game_id:
            raise StageError(f"Twitch game {cfg.twitch.game_name!r} could not be found")
        game_ids = [game_id]
    else:
        games = _twitch_top_games(cfg, token, first=10)
        if not games:
            raise StageError("Twitch API returned no top games")
        game_ids = [g["id"] for g in games]
        log.info("Querying top %d Twitch games for clips", len(game_ids))

    per_game = max(1, cfg.discovery.max_candidates // max(len(game_ids), 1))
    all_clips: list[dict[str, Any]] = []
    for game_id in game_ids:
        clips = _twitch_clips(cfg, token, game_id, started_at, ended_at, cfg.twitch.language, per_game)
        log.info("Twitch game %s -> %d clips", game_id, len(clips))
        all_clips.extend(clips)

    candidates: list[Candidate] = []
    for clip in all_clips:
        clip_id = clip.get("id") or ""
        if not clip_id:
            continue
        candidates.append(
            Candidate(
                video_id=str(clip_id),
                url=f"https://clips.twitch.tv/{clip_id}",
                title=clip.get("title") or "Untitled clip",
                channel=clip.get("broadcaster_name") or "Unknown",
                channel_id=clip.get("broadcaster_id") or "",
                views=int(clip.get("view_count") or 0),
                likes=0,
                comments=0,
                duration_sec=float(clip.get("duration") or 0.0),
                published_at=clip.get("created_at") or "",
                license="twitch",
                platform="twitch",
                metadata_complete=True,
            )
        )

    return _dedupe(candidates)


# ---------------------------------------------------------------------------
# Instagram sources
# ---------------------------------------------------------------------------

APIFY_API_BASE = "https://api.apify.com/v2"


def _apify_get(url: str, token: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    """Make an authenticated Apify API GET call."""
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("`requests` is required for the instagram apify backend") from exc

    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    if resp.status_code in (401, 403):
        raise StageError(
            f"Apify API rejected the request (HTTP {resp.status_code}). "
            "Check INSTAGRAM_API_KEY / instagram.api_key."
        )
    resp.raise_for_status()
    return resp.json()


def _apify_run_actor(cfg: Config, token: str) -> str:
    """Start an Apify actor run and return the dataset ID once it finishes."""
    try:
        import requests  # noqa: PLC0415
    except ImportError as exc:
        raise StageError("`requests` is required for the instagram apify backend") from exc

    insta = cfg.instagram
    actor_id = insta.actor_id
    if not actor_id:
        raise StageError("instagram.backend is 'apify' but instagram.actor_id is not set")

    run_url = f"{APIFY_API_BASE}/acts/{actor_id}/runs"
    input_body: dict[str, Any] = {
        "maxResults": min(insta.max_results, 100),
    }
    # Target music-video hashtags first, then free-form search terms.
    hashtags = [h for h in (insta.hashtags or []) if h]
    if hashtags:
        input_body["hashtags"] = hashtags
    queries = [q for q in (insta.queries or []) if q]
    if queries:
        input_body["search"] = queries
    if not hashtags and not queries:
        log.warning("No instagram.hashtags or instagram.queries configured; actor may use defaults")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    resp = requests.post(run_url, headers=headers, json=input_body, timeout=60)
    if resp.status_code in (401, 403):
        raise StageError(
            f"Apify API rejected the request (HTTP {resp.status_code}). "
            "Check INSTAGRAM_API_KEY / instagram.api_key."
        )
    resp.raise_for_status()
    data = resp.json()
    run_id = data.get("data", {}).get("id")
    if not run_id:
        raise StageError("Apify run response contained no run id")

    # Poll until the run finishes (succeeded/failed/aborted).
    status_url = f"{APIFY_API_BASE}/actor-runs/{run_id}"
    import time  # noqa: PLC0415
    for _ in range(30):
        status_data = _apify_get(status_url, token)
        status = status_data.get("data", {}).get("status")
        if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
            if status != "SUCCEEDED":
                raise StageError(f"Apify actor run {run_id} ended with status {status}")
            return str(status_data.get("data", {}).get("defaultDatasetId", ""))
        time.sleep(2)
    raise StageError(f"Apify actor run {run_id} did not finish within the polling window")


def _apify_dataset_items(dataset_id: str, token: str, max_results: int) -> list[dict[str, Any]]:
    """Fetch items from an Apify dataset."""
    url = f"{APIFY_API_BASE}/datasets/{dataset_id}/items"
    params = {"limit": min(max_results, 100)}
    data = _apify_get(url, token, params=params, timeout=60)
    return data if isinstance(data, list) else []


def _apify_to_candidates(items: list[dict[str, Any]]) -> list[Candidate]:
    """Map Apify Instagram scraper output to Candidates."""
    candidates: list[Candidate] = []
    for item in items:
        url = item.get("url") or item.get("link") or item.get("shortCode")
        if not url:
            continue
        if not url.startswith("http"):
            url = f"https://www.instagram.com/reel/{url}/"
        # Try to extract a stable id from the URL.
        match = re.search(r"/reel/([^/?]+)", url)
        video_id = match.group(1) if match else url
        candidates.append(
            Candidate(
                video_id=str(video_id),
                url=url,
                title=item.get("caption") or item.get("title") or "Instagram reel",
                channel=item.get("ownerUsername") or item.get("username") or "Unknown",
                channel_id=item.get("ownerId") or "",
                views=int(item.get("likesCount") or item.get("videoViewCount") or item.get("viewCount") or 0),
                likes=int(item.get("likesCount") or 0),
                comments=int(item.get("commentsCount") or 0),
                duration_sec=float(item.get("videoDuration") or 0.0),
                published_at=item.get("timestamp") or "",
                license="instagram",
                platform="instagram",
                metadata_complete=True,
            )
        )
    return candidates


def _instagram_manual(cfg: Config) -> list[Candidate]:
    """Read Instagram Reels URLs from the configured sources file."""
    path = cfg.root / cfg.instagram.sources_file
    if not path.exists():
        raise StageError(f"instagram.backend is 'manual' but {path} does not exist")

    candidates: list[Candidate] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        match = re.search(r"/reel/([^/?]+)", url)
        video_id = match.group(1) if match else url
        candidates.append(
            Candidate(
                video_id=str(video_id),
                url=url,
                title=f"manual:{video_id}",
                channel="manual",
                license="instagram",
                platform="instagram",
            )
        )
    return candidates


def from_instagram(cfg: Config) -> list[Candidate]:
    """Discover Instagram Reels/clip candidates that contain music + video.

    Supports two backends:
      - manual: read URLs from instagram.sources_file
      - apify: run a configured Apify Instagram scraper actor, filtered to
        music-video hashtags/search terms

    Static-photo posts are out of scope; the backend configuration should
    request Reels/video results.
    """
    insta = cfg.instagram
    token = insta.api_key or cfg.instagram_api_key

    if insta.backend == "manual":
        return _instagram_manual(cfg)

    if insta.backend == "apify":
        if not token:
            raise StageError(
                "instagram.backend is 'apify' but no API key is set. "
                "Add INSTAGRAM_API_KEY to .env or set instagram.api_key."
            )
        dataset_id = _apify_run_actor(cfg, token)
        if not dataset_id:
            raise StageError("Apify run succeeded but returned no dataset id")
        items = _apify_dataset_items(dataset_id, token, insta.max_results)
        log.info("Apify returned %d Instagram items", len(items))
        return _dedupe(_apify_to_candidates(items))

    raise StageError(f"Unknown instagram.backend {insta.backend!r}; expected 'manual' or 'apify'")


# ---------------------------------------------------------------------------
# Filtering / ranking
# ---------------------------------------------------------------------------

def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """First occurrence of each video id wins; search order is roughly relevance."""
    seen: dict[str, Candidate] = {}
    for c in candidates:
        seen.setdefault(c.video_id, c)
    return list(seen.values())


def apply_filters(
    candidates: list[Candidate], cfg: Config, *, check_license: bool = True
) -> list[Candidate]:
    """Drop candidates that fail the configured limits.

    ``check_license=False`` runs everything except the license test, for callers
    that have not looked up licenses yet (see :func:`_hydrate_metadata`).
    """
    d = cfg.discovery
    blocked_channels = {c.lower() for c in d.blocked_channels}
    blocked_keywords = [k.lower() for k in d.blocked_keywords]

    kept: list[Candidate] = []
    dropped: dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for c in candidates:
        title = c.title.lower()
        if (c.views or c.metadata_complete) and c.views < d.min_views:
            drop("below min_views")
        elif c.duration_sec and not (d.min_duration_sec <= c.duration_sec <= d.max_duration_sec):
            drop("duration out of range")
        elif c.channel.lower() in blocked_channels or c.channel_id in d.blocked_channels:
            drop("blocked channel")
        elif any(k in title for k in blocked_keywords):
            drop("blocked keyword")
        elif (
            check_license
            and d.license_filter == "creativeCommon"
            and c.platform not in ("twitch", "instagram")
            and c.license != "creativeCommon"
        ):
            # Twitch and Instagram clips are never Creative Commons; applying
            # the CC filter would silently empty those pools.
            drop("not creative-commons licensed")
        else:
            kept.append(c)

    for reason, count in sorted(dropped.items()):
        log.info("Filtered out %3d candidates: %s", count, reason)
    return kept


def rank_and_select(candidates: list[Candidate], cfg: Config) -> list[Candidate]:
    """Score, dedupe by video and channel, and take the top N."""
    unique: dict[str, Candidate] = {}
    for c in candidates:
        c.score, c.reasons = virality_score(c)
        existing = unique.get(c.video_id)
        if existing is None or c.score > existing.score:
            unique[c.video_id] = c

    ordered = sorted(unique.values(), key=lambda c: c.score, reverse=True)

    # At most two clips per channel keeps the compilation feeling varied.
    per_channel: dict[str, int] = {}
    selected: list[Candidate] = []
    for c in ordered:
        key = c.channel_id or c.channel
        if per_channel.get(key, 0) >= 2:
            continue
        per_channel[key] = per_channel.get(key, 0) + 1
        selected.append(c)
        if len(selected) >= cfg.discovery.max_results:
            break
    return selected


def discover(cfg: Config | None = None) -> list[Candidate]:
    cfg = cfg or load_config()
    cfg.make_dirs()

    sources = {
        "youtube_api": from_youtube_api,
        "ytdlp_search": from_ytdlp_search,
        "music_video": from_music_video,
        "file": from_file,
        "twitch": from_twitch_api,
        "instagram": from_instagram,
    }
    source_name = cfg.discovery.source
    if source_name not in sources:
        raise StageError(f"Unknown discovery.source {source_name!r}; expected one of {sorted(sources)}")

    if source_name == "twitch":
        scope = cfg.twitch.game_name or "top games"
        log.info("Discovering viral candidates via %s (%s, period=%s)", source_name, scope, cfg.twitch.period)
    elif source_name == "music_video":
        log.info("Discovering viral music video candidates via %s (region=%s)", source_name, cfg.discovery.region)
    elif source_name == "instagram":
        scope = cfg.instagram.backend
        if cfg.instagram.hashtags:
            scope = f"{scope}, hashtags={cfg.instagram.hashtags}"
        log.info("Discovering Instagram music video candidates via %s (%s)", source_name, scope)
    else:
        log.info("Discovering viral candidates via %s (region=%s)", source_name, cfg.discovery.region)
    try:
        raw = sources[source_name](cfg)
    except StageError:
        if source_name != "youtube_api":
            raise
        log.warning("YouTube Data API unavailable - falling back to keyless yt-dlp search")
        raw = from_ytdlp_search(cfg)

    log.info("%d candidates from %s", len(raw), source_name)
    selected = rank_and_select(apply_filters(raw, cfg), cfg)

    if not selected:
        hint = "Loosen discovery.min_views / discovery.max_duration_sec, or add more search_queries."
        if cfg.discovery.license_filter == "creativeCommon":
            hint = (
                "The Creative Commons pool is much smaller than the general one, and "
                f"discovery.min_views is {cfg.discovery.min_views:,}. Lower it (5000 is a "
                "realistic starting point for CC clips), widen the duration range, or add "
                "search_queries."
            )
        raise StageError(f"No candidates survived filtering. {hint}")

    for i, c in enumerate(selected, 1):
        log.info("#%d  %-6.1f  %-45.45s  [%s]", i, c.score, c.title, ", ".join(c.reasons[:2]))

    out = cfg.path("workspace") / "candidates.json"
    write_json(out, [asdict(c) for c in selected])
    log.info("Wrote %d candidates to %s", len(selected), out)
    return selected


def load_candidates(cfg: Config) -> list[Candidate]:
    path = cfg.path("workspace") / "candidates.json"
    data = read_json(path)
    if not data:
        raise StageError(f"{path} not found or empty - run the discover stage first")
    return [Candidate(**item) for item in data]


if __name__ == "__main__":
    discover()
