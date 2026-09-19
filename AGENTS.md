# AGENTS.md

Guide for AI coding agents (and humans) working in this repository.

This project discovers videos that are going viral, cuts the best few seconds
out of each one, and assembles them into a single vertical compilation with
burned-in captions and a credits card.

---

## 1. The pipeline agents

The pipeline is six single-responsibility stages. Each is a module in
`scripts/`, each reads the previous stage's JSON output and writes its own, and
each is runnable on its own. Nothing holds state in memory between stages, so
any stage can be re-run without repeating the expensive ones.

```
discover ──► fetch ──► process ──► subtitles ──► editor ──► metadata
   │           │          │            │            │          │
candidates  assets      clips      subtitles     renders    <name>.md
  .json      .json      .json        .json        *.mp4     (upload text)
```

| # | Agent | Module | Responsibility |
|---|-------|--------|----------------|
| 1 | Discovery | [scripts/discover.py](scripts/discover.py) | Find and rank viral candidates |
| 2 | Fetch | [scripts/fetch.py](scripts/fetch.py) | Download sources + attribution metadata |
| 3 | Process | [scripts/process.py](scripts/process.py) | Cut the highlight, normalize the format |
| 4 | Subtitles | [scripts/subtitles.py](scripts/subtitles.py) | Transcribe into SRT + styled ASS |
| 5 | Editor | [scripts/editor.py](scripts/editor.py) | Burn overlays, add cards, concatenate |
| 6 | Metadata | [scripts/metadata.py](scripts/metadata.py) | Title, description, tags, credits |

Supporting modules — not stages:

| Module | Role |
|--------|------|
| [scripts/config.py](scripts/config.py) | Typed config: dataclasses ← `config.yaml` ← `.env` |
| [scripts/ffmpeg_utils.py](scripts/ffmpeg_utils.py) | Every ffmpeg/ffprobe invocation in the project |
| [scripts/highlight.py](scripts/highlight.py) | Which seconds of a video are worth keeping |
| [scripts/llm_client.py](scripts/llm_client.py) | Shared OpenAI-compatible chat client |
| [scripts/curator.py](scripts/curator.py) | Optional LLM ranking + highlight hints |
| [scripts/util.py](scripts/util.py) | Logging, atomic JSON I/O, slugs, SRT timestamps |
| [main.py](main.py) | CLI orchestrator and flag → config overrides |

---

### Agent 1 — Discovery (`scripts/discover.py`)

**In:** config only. **Out:** `workspace/candidates.json`, `list[Candidate]`.

Four interchangeable sources, chosen by `discovery.source`:

- The `twitch` source calls the Helix `/games` and `/clips` endpoints. Clips are
  ranked by the same `virality_score()` as YouTube candidates (velocity,
  engagement, freshness). Twitch clips carry `platform="twitch"`, which the fetch
  stage uses to land downloads in `workspace/downloads_twitch/` instead of
  `workspace/downloads/`.
- The `music_video` source targets the YouTube Music category (`videoCategoryId=10`)
  and music-video-themed queries. Every search includes a visual term ("video",
  "shorts", "clip") so results are videos with music, never audio-only tracks.
  Candidates carry `platform="music_video"` and are downloaded to
  `workspace/downloads_music_video/`.
- The `instagram` source supports a `manual` URL file or an `apify` third-party
  backend configured with music-video hashtags/search terms. Candidates carry
  `platform="instagram"` and are downloaded to `workspace/downloads_instagram/`.

| Source | Needs | Signals available |
|--------|-------|-------------------|
| `youtube_api` | `YOUTUBE_API_KEY` | views, likes, comments, publish time, licence |
| `ytdlp_search` | nothing | views, duration only (see licence lookup below) |
| `music_video` | `YOUTUBE_API_KEY` (optional) | same as YouTube sources |
| `file` | `sources.txt` | none — you picked the clips |
| `twitch` | `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET` | views, duration, broadcaster, created_at |
| `instagram` | `INSTAGRAM_API_KEY` (for `apify`) | views/likes, duration, username, timestamp |

**Licences.** A flat yt-dlp search result has no `license` field, so candidates
default to `"youtube"` and a `creativeCommon` filter would reject the whole
pool. When that filter is on, the `ytdlp_search` and `file` sources call
`_hydrate_metadata()`, which does one `extract_info(..., process=False)` per
video to fill in licence, upload date and engagement counts; `ytdlp_search`
additionally searches through `CC_SEARCH_URL` (YouTube's own CC filter) and
pre-filters on the cheap fields first, so the ~1s-per-video lookup is only paid
on plausible clips. yt-dlp's human-readable licence string and the API's slug
are both put through `normalise_license()` — compare against `"creativeCommon"`,
never against the raw value.

`youtube_api` automatically falls back to `ytdlp_search` when the key is absent
or the quota is spent. It queries two endpoints: the regional `mostPopular`
chart, and one `search.list` per configured query, hydrating search hits through
a single batched `videos.list` because `search.list` returns no statistics.

**Query generation.** If `discovery.llm_query_enabled` is true, discovery asks
the configured OpenAI-compatible endpoint to generate a fresh batch of search
queries from `discovery.llm_query_theme` instead of replaying the static
`search_queries` list. This is the intended fix for "discovery always returns the
same videos": the LLM varies phrasing and mixes genres (fails, satisfying,
unexpected, skills, caught-on-camera, etc.) every run. When generation fails or
the setting is off, the stage falls back to `search_queries`.

**Ranking.** `virality_score()` deliberately does not rank by raw view count —
that just resurfaces old hits. It blends:

- **velocity** — `views / hours_since_upload`, log-scaled, the dominant term;
- **engagement** — `(likes + 3×comments) / views`, capped at 20%;
- **freshness** — exponential decay with a ~1 week half-life.

Then `rank_and_select()` dedupes by video ID and caps each channel at two clips
so a compilation is not one creator's back catalogue.

**Known limitation:** flat yt-dlp search results carry no upload date, so
`age_hours` defaults to 30 days and velocity degrades to a function of views.
This is expected; use the API source when ranking quality matters. (The
`creativeCommon` path is the exception — hydration gives it real dates.)

`Candidate.metadata_complete` records whether the counts came from such a
lookup. `min_views` is only applied to a zero view count when it is set, so an
unreported count is never mistaken for an unpopular video.

---

### Agent 1.5 — Curator (`scripts/curator.py`) *(optional)*

**In:** `workspace/candidates.json`, config. **Out:** `workspace/curated_candidates.json`, `list[CuratedCandidate]`.

An optional LLM stage between Discovery and Fetch. It reviews the top
`curator.max_candidates` discovery results, fetches short descriptions and the
first ~30 seconds of English transcript per video, and asks an OpenAI-compatible
chat endpoint (Ollama, OpenAI, LocalAI, etc.) to:

1. select the best `discovery.max_results` clips for the compilation theme,
2. give a one-sentence reason per clip,
3. assign a confidence score 1-10,
4. suggest a `suggested_start_sec` inside the safe bounds.

The response is parsed strictly as JSON. If the endpoint is unreachable, the
response is malformed, or parsing fails, the stage falls back to the discovery
ranking unchanged.

The suggested start time is carried on `Asset.llm_suggested_start` and then on
`Clip` via `process.py`. `highlight.find_highlight()` uses it as a soft anchor:
if present and inside the media bounds, the clip window starts there while
keeping the configured `clip_seconds` length.

---

### Agent 2 — Fetch (`scripts/fetch.py`)

**In:** `candidates.json`. **Out:** `workspace/downloads/*.mp4`, `assets.json`.

Downloads with yt-dlp's Python API. Files are named `<video_id>__<slug>.mp4`, so
`_existing_download()` can find and reuse them — re-running the stage never
re-downloads. Raw yt-dlp metadata is written to `<video_id>.info.json` next to
the media; the credits card and description are built from it, which is why
`uploader`, `uploader_url`, `license` and `webpage_url` are preserved on `Asset`.

A single failed download is logged and skipped, not fatal. All downloads failing
is fatal. If YouTube starts demanding sign-in, set
`download.cookies_from_browser: chrome` in `config.yaml`.

---

### Agent 3 — Process (`scripts/process.py` + `scripts/highlight.py`)

**In:** `assets.json`. **Out:** `workspace/clips/NN_<id>.mp4`, `clips.json`.

Two jobs:

**Pick the moment.** `highlight.find_highlight()` offers three strategies:

- `audio_energy` *(default)* — decodes the audio to 16 kHz mono PCM, takes RMS in
  250 ms frames, convolves with a clip-length box kernel, and returns the loudest
  sustained window. Sustained, not peak: a single transient is a door slam, a
  sustained rise is the payoff. Falls back to a fixed offset when numpy is
  missing or the source is silent.
- `scene` — the window containing the most ffmpeg-detected cuts.
- `start` — the opening, after `skip_intro_sec`.

`skip_intro_sec`/`skip_outro_sec` keep intros, sponsor reads and outro cards out
of the search range. The search bounds are clamped, so a source shorter than
`clip_seconds` yields the whole file instead of crashing.

**Normalize.** Every clip is re-encoded to the exact `render.*` spec: same
resolution, fps, pixel format, SAR, audio sample rate, channel layout, and
loudness (`loudnorm` to `-14 LUFS`, YouTube's target). This is what lets the
editor concatenate by stream copy later. Sources with no audio track get
synthesised silence so every part has identical stream layout.

`render.fit` controls reframing: `blur_pad` (whole frame over a blurred,
darkened backdrop — the default), `crop` (fill, lose the edges), `pad` (flat
bars).

---

### Agent 4 — Subtitles (`scripts/subtitles.py`)

**In:** `clips.json`. **Out:** `workspace/subtitles/*.srt`, `*.ass`, `subtitles.json`.

Engine is auto-detected: `faster-whisper` preferred (~4× faster on CPU at int8),
`openai-whisper` accepted, and if neither is installed the stage logs a warning
and returns empty — the pipeline continues without captions rather than dying.

Audio is extracted to 16 kHz mono WAV first so Whisper never decodes video.

`group_into_cues()` re-chunks word-level timings into short cues, breaking on
character limit, a 0.7 s pause, or 3.5 s elapsed, then trims overlaps. Whole
Whisper segments are frequently 15+ words and unreadable as burned-in captions.

**Two files per clip, on purpose:**

- `.srt` — portable sidecar; upload it as a caption track.
- `.ass` — what actually gets burned in.

The `.ass` exists because of a real trap: ffmpeg's SRT→ASS converter hard-codes
the script canvas at **384×288**, and libass interprets `force_style`'s
`FontSize`/`MarginV` in *that* space before scaling to the frame. On a 1080×1920
render that is a 6.67× multiplier — `MarginV: 260` becomes 1733 px and the
captions leave the top of the picture. Writing our own `.ass` with explicit
`PlayResX`/`PlayResY` makes `captions.font_size` and `captions.margin_v` honest
pixels. `build_subtitles_filter()` still handles `.srt` by dividing both values
by that ratio.

---

### Agent 5 — Editor (`scripts/editor.py`)

**In:** `clips.json` + `subtitles.json`. **Out:** `output/compilation-<stamp>.mp4`.

Assembles `[title card] → [clip 1 … clip N] → [credits card]`.

Per clip, one overlay pass burns in: the subtitle track, a `#N` badge that
auto-hides after 2.2 s so it never covers the action, and the optional
watermark. Video is re-encoded; **audio is stream-copied** — it was already
loudness-normalized upstream and must not be normalized twice.

The join uses ffmpeg's concat demuxer with `-c copy`. That is only valid because
every part shares an encode spec — see the invariant below.

Text overlays are passed through `drawtext`'s `textfile=` with `expansion=none`.
Titles come from the internet and would otherwise need escaping for `:`, `,`,
`\`, `'` and `%` expansion.

---

### Agent 6 — Metadata (`scripts/metadata.py`)

**In:** `list[Clip]`. **Out:** `output/compilation-<stamp>.md`.

Title (truncated to YouTube's 100 characters), tag list from title keyword
frequency, chapter timestamps offset by the title card, and a numbered credits
block naming every creator, channel URL, source URL and licence.

---

## 2. Invariants — break these and the pipeline breaks

1. **Concat requires identical encode parameters.** The process stage forces
   resolution, fps, pixel format, SAR, audio rate and channel layout for exactly
   this reason. Any new stage that produces a part for `concat()` must use
   `ffmpeg_utils.encode_args()`. If you must diverge, switch to the concat
   *filter* and re-encode.
2. **Audio is normalized exactly once**, in the process stage. Overlay passes
   use `-c:a copy`.
3. **Every ffmpeg call goes through `scripts/ffmpeg_utils.py`.** Do not add bare
   `subprocess.run(["ffmpeg", ...])` elsewhere; `run()` centralises error
   reporting and surfaces the tail of stderr in `StageError`.
4. **Attribution survives every stage.** `channel`, `channel_url`, `source_url`
   and `license` are carried on `Candidate` → `Asset` → `Clip`. Do not drop them
   when adding fields.
5. **Stage state is JSON in `workspace/`**, written atomically via
   `util.write_json()`. Keep stages resumable; do not pass objects that only
   exist in memory.
6. **Config is the single source of tunables.** Add a field to the dataclass in
   `scripts/config.py` *and* to `config.yaml`. Unknown YAML keys log a warning
   rather than failing, so a typo is silent — check the log.

---

## 3. Conventions

- Python 3.12, `from __future__ import annotations`, PEP 604 (`str | None`).
- Dataclasses for anything crossing a stage boundary.
- `util.get_logger(name)` for output. No bare `print()` except the CLI notice.
- Raise `util.StageError` for expected failures with an actionable message;
  `main.py` turns it into exit code 1 without a traceback.
- Optional dependencies (`yaml`, `requests`, `numpy`, `yt_dlp`, whisper) are
  imported **inside** the function that needs them, so an unrelated stage still
  runs when one is missing.
- Comments explain *why*, especially around ffmpeg's sharp edges. The codebase
  is deliberately comment-heavy at those points.

---

## 4. Setup and commands

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
sudo apt install ffmpeg          # ffmpeg + ffprobe are required
cp .env.example .env             # optional: add YOUTUBE_API_KEY
```

```bash
python main.py run                       # full pipeline
python main.py run --limit 4 --clip-seconds 6
python main.py run --landscape --no-captions
python main.py run --source file          # only clips listed in sources.txt
python main.py run --source twitch        # Twitch clips via Helix API
python main.py run --source music_video   # viral music VIDEO clips from YouTube
python main.py run --source instagram     # Instagram music-video Reels (manual or Apify)
python main.py run --creative-commons     # CC-licensed sources only

python main.py discover                  # stages individually
python main.py fetch
python main.py process
python main.py subtitles
python main.py edit                      # re-assemble from cached clips
python main.py clean                     # wipe workspace/, keep output/
python main.py -v <command>              # debug logging
```

Note: this repo's `.venv/bin/pip` shebang points at a stale path, so console
scripts fail. Use `.venv/bin/python -m pip …`, or recreate the venv.

---

## 5. Testing changes

There is no test suite. The reliable check is a synthetic end-to-end run — it
needs no network and no API key:

```bash
# three 30s sources with a loud region planted at a known timestamp
ffmpeg -f lavfi -i "testsrc2=size=640x360:rate=30:duration=30" \
       -f lavfi -i "sine=frequency=440:duration=30" \
       -filter_complex "[1:a]volume=0.03,volume=enable='between(t,18,26)':volume=25[a]" \
       -map 0:v -map "[a]" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest fake1.mp4
```

Construct `Asset` objects pointing at those files, call `process.process()` then
`editor.build()`, and assert that:

- the chosen window covers the planted loud region (highlight detection works);
- final duration ≈ title card + Σ clip durations + outro (concat worked);
- final resolution equals `render.width × render.height`.

**Always extract a frame and look at it.** Overlay bugs — captions off-frame,
badges colliding with text, watermarks under the safe area — are invisible in
logs and obvious in a single PNG:

```bash
ffmpeg -ss 3.5 -i output/compilation-*.mp4 -frames:v 1 frame.png
```

When touching discovery, `python main.py discover` is safe on its own: it reads
metadata only and downloads nothing.

---

## 6. Rights and licensing

Every clip belongs to whoever filmed it. Downloading from YouTube is against its
Terms of Service, and re-uploading someone else's footage can be copyright
infringement regardless of a credits card — "transformative use" is a legal test
a court applies, not a setting in `config.yaml`. Compilation channels do get
struck and demonetised for exactly this.

The pipeline does what it is told; the safer paths are built in and worth using:

- `discovery.license_filter: creativeCommon` — restricts to CC-BY videos, which
  are licensed for reuse *with attribution*. Keep `branding.outro_credits: true`.
- `discovery.source: file` with `sources.txt` — you clear every clip yourself.
- Keep the generated `.md` credits block in the upload description either way.

Do not remove the notice printed by `main.py run`, the credits card, or the
attribution fields on the dataclasses.
