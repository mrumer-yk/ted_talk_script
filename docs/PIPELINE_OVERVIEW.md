# Pipeline Overview

This document describes the end-to-end TED Talk filtering pipeline that downloads videos from the official TED channel, clips them, filters for single-speaker segments using HOG person detection, and saves one high-quality clip per speaker.

## Goals
- Only download from official TED channel (@TED)
- Exclude Shorts (< 3 minutes)
- Exclude subtitle/hardcoded-caption videos
- Avoid duplicates (skip already processed speakers)
- Keep one best clip per speaker (single speaker, minimal audience)
- Parallelize per-speaker filtering for throughput

## High-level Flow
1. Fetch latest videos from @TED using yt-dlp flat playlist.
2. Skip titles indicating subtitles; enforce duration > 180 seconds.
3. Extract speaker name from title and sanitize to folder-name.
4. Skip if speaker already exists in `ted_clips_new/` or `filtered_videos_2/`.
5. Download the video with yt-dlp.
6. Stage 1: Clip into 5×30s segments via `tool.py`.
7. Stage 2: Launch `process_single_speaker.py` (up to 4 in parallel) to evaluate clips with HOG.
8. Select the best single-speaker clip and copy to `filtered_videos_2/`.
9. Repeat until target count (e.g., 2000) is reached.

## Key Scripts
- `batch_ted_clipper.py` — orchestrator; fetches, downloads, clips, and dispatches filtering.
- `tool.py` — video utilities: clipping, frame sampling, HOG helper functions.
- `process_single_speaker.py` — per-speaker filter runner (enables parallelism).
- `ted_video_filter.py` — core second-stage filtering logic and scoring.
- `filter_config.py` — thresholds, directories, parallelism.
- `run_filter.py` — optional batch runner for existing `ted_clips_new/` folders.

## Important Directories
- `ted_clips_new/` — stage-1 output (raw clips), per speaker.
- `filtered_videos_2/` — final results (one clip per speaker).

## Duplicate Avoidance
- Before downloading, the orchestrator checks sanitized speaker names against both `ted_clips_new/` and `filtered_videos_2/` and skips duplicates.

## Shorts & Subtitles Exclusion
- Shorts excluded by yt-dlp filter: `--match-filter "duration > 180"`.
- Subtitle-like titles (e.g., "subtitles", "captions", "cc") are skipped.
- Subtitle files are not written: `--no-write-subs --no-write-auto-subs`.
