#!/usr/bin/env python3
"""Orchestrator for the viral compilation pipeline.

    python main.py run                    # full pipeline
    python main.py run --limit 4          # fewer clips
    python main.py discover               # one stage at a time
    python main.py edit                   # re-assemble from cached clips

Every stage writes its result into workspace/*.json, so any stage can be re-run
on its own without repeating the expensive ones.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

from scripts.config import load_config
from scripts.util import StageError, get_logger, setup_logging

log = get_logger("main")

USAGE_NOTICE = """\
Reminder: downloaded clips belong to whoever made them. Re-uploading someone
else's footage without permission can breach copyright and YouTube's terms,
whatever the pipeline makes possible. For a channel you intend to monetize,
either set discovery.license_filter: creativeCommon (and keep the credits card),
or use discovery.source: file with clips you have cleared."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Discover viral videos and cut them into a new compilation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_NOTICE,
    )
    parser.add_argument("-c", "--config", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run every stage end to end")
    run.add_argument("--limit", type=int, help="number of clips in the compilation")
    run.add_argument("--clip-seconds", type=float, help="length of each clip")
    run.add_argument("--source", choices=["youtube_api", "ytdlp_search", "music_video", "file", "twitch", "instagram"],
                     help="where candidates come from")
    run.add_argument("--region", help="two-letter region code for trending (e.g. US, IN, GB)")
    run.add_argument("--query", action="append", dest="queries",
                     help="search query; repeatable, replaces the configured list")
    run.add_argument("--fit", choices=["blur_pad", "crop", "pad"], help="how clips fill the frame")
    run.add_argument("--landscape", action="store_true", help="render 1920x1080 instead of 9:16")
    run.add_argument("--no-captions", action="store_true", help="skip transcription")
    run.add_argument("--creative-commons", action="store_true",
                     help="only consider Creative Commons licensed videos")

    for name, help_text in [
        ("discover", "find and rank viral candidates"),
        ("curate", "optionally refine candidates with an LLM"),
        ("fetch", "download the selected candidates"),
        ("process", "cut highlights and normalize them"),
        ("subtitles", "transcribe clips into SRT"),
        ("edit", "assemble the final compilation"),
    ]:
        stage = sub.add_parser(name, help=help_text)
        stage.add_argument("--limit", type=int, help="cap the number of items")
        if name == "discover":
            # Discovery flags belong on the stage too, so the pool can be tuned
            # without re-running the download and render stages each time.
            stage.add_argument("--source", choices=["youtube_api", "ytdlp_search", "music_video", "file", "twitch", "instagram"],
                               help="where candidates come from")
            stage.add_argument("--region", help="two-letter region code (e.g. US, IN, GB)")
            stage.add_argument("--query", action="append", dest="queries",
                               help="search query; repeatable, replaces the configured list")
            stage.add_argument("--creative-commons", action="store_true",
                               help="only consider Creative Commons licensed videos")

    sub.add_parser("clean", help="delete the workspace (keeps output/)")
    return parser


def apply_overrides(cfg, args: argparse.Namespace) -> None:
    """CLI flags win over config.yaml."""
    if getattr(args, "limit", None):
        cfg.discovery.max_results = args.limit
    if getattr(args, "clip_seconds", None):
        cfg.highlight.clip_seconds = args.clip_seconds
    if getattr(args, "source", None):
        cfg.discovery.source = args.source
    if getattr(args, "region", None):
        cfg.discovery.region = args.region.upper()
    if getattr(args, "queries", None):
        cfg.discovery.search_queries = args.queries
        cfg.discovery.llm_query_enabled = False
    if getattr(args, "fit", None):
        cfg.render.fit = args.fit
    if getattr(args, "landscape", False):
        cfg.render.width, cfg.render.height = 1920, 1080
        cfg.captions.margin_v = 90
    if getattr(args, "no_captions", False):
        cfg.captions.enabled = False
    if getattr(args, "creative_commons", False):
        cfg.discovery.license_filter = "creativeCommon"


def cmd_run(cfg) -> int:
    from scripts import curator, discover, editor, fetch, process, subtitles

    started = time.time()
    print(USAGE_NOTICE, file=sys.stderr)
    print("-" * 72, file=sys.stderr)

    log.info("[1/6] Discovering viral candidates")
    candidates = discover.discover(cfg)

    log.info("[2/6] Curating candidates with LLM")
    candidates = curator.curate(cfg, candidates)

    log.info("[3/6] Downloading %d videos", len(candidates))
    assets = fetch.fetch(cfg, candidates)

    log.info("[4/6] Generating captions for %d downloaded videos", len(assets))
    subtitles.subtitle_assets(cfg, assets)

    # log.info("[5/6] Cutting highlights")
    # clips = process.process(cfg, assets)

    # log.info("[6/6] Assembling final video")
    # output = editor.build(cfg, clips)

    log.info("Done in %.1fs -> downloaded %d videos with subtitles to %s", time.time() - started, len(assets), cfg.path("downloads"))
    return 0


def cmd_clean(cfg) -> int:
    workspace = cfg.path("workspace")
    if workspace.exists():
        shutil.rmtree(workspace)
        log.info("Removed %s", workspace)
    else:
        log.info("Nothing to clean")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    cfg.make_dirs()

    try:
        if args.command == "run":
            return cmd_run(cfg)
        if args.command == "clean":
            return cmd_clean(cfg)

        from scripts import curator, discover, editor, fetch, process, subtitles

        stages = {
            "discover": lambda: discover.discover(cfg),
            "curate": lambda: curator.curate(cfg),
            "fetch": lambda: fetch.fetch(cfg),
            "process": lambda: process.process(cfg),
            "subtitles": lambda: subtitles.subtitles(cfg),
            "edit": lambda: editor.build(cfg),
        }
        stages[args.command]()
        return 0
    except StageError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.warning("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
