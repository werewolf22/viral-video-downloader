# Viral compilation pipeline

Finds videos that are going viral, cuts the best few seconds out of each one,
and assembles them into a single vertical compilation with burned-in captions,
a title card and a credits card.

```
discover ──► fetch ──► process ──► subtitles ──► editor ──► metadata
  rank      download   cut the     transcribe    burn in    title, tags,
 by speed   + credits  loud part   into cues     + join     credits
```

Output: `output/compilation-<timestamp>.mp4` plus a matching `.md` with the
title, tags, chapter timestamps and credits ready to paste into a description.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
sudo apt install ffmpeg            # ffmpeg + ffprobe are required
```

Optional credentials:

```bash
cp .env.example .env               # add YouTube Data API v3 and/or Twitch keys
```

- **YouTube Data API v3** (`YOUTUBE_API_KEY`) — enables `discovery.source: youtube_api`
  with real like/comment counts and upload times. Without a key the default
  `ytdlp_search` source works fine, just with fewer signals to rank on.
- **Twitch Helix API** (`TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`) — enables
  `discovery.source: twitch` for trending Twitch clips. Register an app at
  https://dev.twitch.tv/console/apps/create.
- **YouTube music VIDEO clips** — no extra key needed; uses the same
  `YOUTUBE_API_KEY` if present, otherwise falls back to `ytdlp_search`. Searches
  always include "video"/"shorts" so results are visual clips with music.
- **Instagram music VIDEO clips** — no free viral API. Use `instagram.backend: manual`
  with a file of Reels URLs, or `apify` with an Apify actor and `INSTAGRAM_API_KEY`.
  Configure music-video hashtags/search terms so results are Reels with music.

## Run

```bash
python main.py run                       # full pipeline
python main.py run --limit 4             # 4 clips instead of 6
python main.py run --clip-seconds 6
python main.py run --landscape           # 1920x1080 instead of 9:16
python main.py run --no-captions         # skip transcription
python main.py run --creative-commons    # only CC-licensed sources
python main.py run --source twitch        # Twitch clips instead of YouTube
python main.py run --source music_video   # viral music VIDEO clips from YouTube
python main.py run --source instagram     # Instagram Reels with music (manual / Apify)
python main.py run --query "skateboard fails" --query "cat fails"
```

Each stage also runs on its own, reading the previous stage's output from
`workspace/`:

```bash
python main.py discover      # rank candidates (metadata only, no downloads)
python main.py fetch
python main.py process
python main.py subtitles
python main.py edit          # re-assemble from cached clips
python main.py clean         # wipe workspace/, keep output/
```

## Configure

Everything lives in [config.yaml](config.yaml); CLI flags override it. The knobs
worth knowing:

| Key | Does |
|-----|------|
| `discovery.source` | `youtube_api` \| `ytdlp_search` \| `music_video` \| `file` \| `twitch` \| `instagram` |
| `discovery.max_results` | How many clips end up in the compilation |
| `discovery.license_filter` | `any`, or `creativeCommon` for reusable sources |
| `discovery.llm_query_enabled` | Let the LLM write fresh search queries every run (stops repeats) |
| `twitch.game_name` | e.g. `"Just Chatting"`; `null` queries top games |
| `twitch.period` | `day` \| `week` \| `month` \| `all` |
| `twitch.language` | Clip language filter, e.g. `"en"` |
| `music_video.genres` | Genres used to build YouTube music-video queries |
| `music_video.queries` | Explicit music-video search queries |
| `music_video.use_shorts` | Append "shorts" to generated music-video queries |
| `instagram.backend` | `manual` (URL file) \| `apify` |
| `instagram.sources_file` | File of Reels URLs for `manual` backend |
| `instagram.actor_id` | Apify actor ID for `apify` backend |
| `instagram.hashtags` | Music-video hashtags to scrape for `apify` backend |
| `instagram.queries` | Extra search terms for Apify actors that support them |
| `curator.enabled` | Let an LLM pick/reorder clips and suggest start times |
| `curator.model` | e.g. `deepseek-v4-flash:cloud`, `llama3.1`, `gpt-4o-mini` |
| `curator.base_url` | OpenAI-compatible endpoint, default `http://localhost:11434/v1` |
| `highlight.clip_seconds` | Length of each clip |
| `highlight.method` | `audio_energy` (loudest moment) \| `scene` \| `start` |
| `render.width/height` | `1080×1920` for Shorts/Reels/TikTok |
| `render.fit` | `blur_pad` \| `crop` \| `pad` |
| `captions.model` | Whisper size: `tiny` … `large-v3` |
| `branding.watermark_text` | e.g. `"@yourchannel"` |

## How it picks the moment

Taking the first N seconds of a video captures the intro and the sponsor read.
Instead, the audio is decoded to PCM and scanned for the loudest *sustained*
window of `clip_seconds` — the laugh, the crash, the crowd — while skipping the
first and last few seconds. Set `highlight.method: scene` to select on cut
density instead.

If `curator.enabled: true`, an LLM also reviews the discovery candidates and
suggests a highlight start time for each clip. The final cut still respects the
configured `clip_seconds`, but it starts at the LLM's suggestion rather than the
pure audio-energy peak. The LLM is optional and falls back to the discovery
ranking when it is unreachable or returns bad JSON.

If `discovery.llm_query_enabled: true` (and the same `curator` block is
configured), the LLM writes a fresh batch of search queries for every `discover`
run instead of replaying the static `search_queries` list. That is the fastest
way to stop the pipeline from returning the same handful of videos every time.

## Rights

Every clip belongs to whoever filmed it. Downloading from YouTube breaches its
Terms of Service, and re-uploading someone else's footage can be copyright
infringement even with credits attached — compilation channels get struck and
demonetised for this routinely.

If the channel matters to you, use one of the safer paths:

- `discovery.license_filter: creativeCommon` — CC-BY videos are licensed for
  reuse with attribution; keep `branding.outro_credits: true`.
- `discovery.source: file` — list URLs you have cleared in `sources.txt`.

The CC pool is far smaller and far less viewed than the general one, so the
default `discovery.min_views: 100000` filters it down to nothing. Drop it to
around `5000` when the license filter is on. Discovery also slows down in this
mode: search results carry no license, so each shortlisted video needs its own
metadata lookup (about a second each) before it can be trusted as CC.

Keep the generated credits block in the upload description either way.

## Instagram setup with Apify

If you want the pipeline to find Instagram Reels for you instead of pasting
URLs manually, the supported path is the **Apify Instagram Scraper** actor.

1. Sign up / log in at https://console.apify.com
2. Go to **Store** and find an Instagram actor, e.g.
   `apify/instagram-scraper` or any Reels scraper that supports `hashtags`
   and/or `search` input.
3. Copy the actor ID from the URL, e.g. `apify/instagram-scraper`.
4. Go to **Settings → Integrations** and create an API token.
5. Add the token to `.env` as `INSTAGRAM_API_KEY=...`
6. In `config.yaml` set:

```yaml
discovery:
  source: instagram
instagram:
  backend: apify
  actor_id: apify/instagram-scraper
  hashtags:
    - viralmusic
    - musicvideo
  queries:
    - trending music reels
  max_results: 50
```

Then run:

```bash
python main.py run --source instagram
```

Note: Apify charges by compute time. A small run is usually a few cents,
but costs depend on the actor and the number of results. The manual backend
(`instagram.backend: manual`) is free and needs no API key.

## Working on the code

See [AGENTS.md](AGENTS.md) for the stage contracts, invariants and the ffmpeg
gotchas worth knowing before editing.
