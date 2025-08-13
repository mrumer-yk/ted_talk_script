# Algorithms

This document explains the core algorithms used in the TED Talk filtering pipeline.

## HOG Person Detection (OpenCV)
- Detector: OpenCV `HOGDescriptor` with default SVM person detector.
- Input: Sampled frames from each clip (e.g., every Nth frame).
- Output: Person bounding boxes per frame.
- Metrics per clip:
  - Single-Person Ratio = frames_with_exactly_one_person / total_sampled_frames
  - Audience Ratio = frames_with_2_or_more_people / total_sampled_frames
- Decision:
  - Keep clips where `single_person_ratio >= MIN_SINGLE_PERSON_RATIO` and `audience_ratio <= MAX_AUDIENCE_RATIO` (set in `filter_config.py`).
  - Among passing clips, select the one with best metrics (highest single-person ratio; tie-break by lowest audience ratio, or other score).

## Speaker Extraction from Title
- Extract a candidate speaker name from the video title (e.g., text before a dash/colon or parentheses patterns).
- Sanitize the name to form a safe folder name (remove invalid characters, trim spaces, unify underscores).
- Use this name consistently across `ted_clips_new/` and `filtered_videos_2/` for duplicate checks.

## Parallel Per-Speaker Filtering
- As soon as a speaker’s 5 clips are created, `process_single_speaker.py` is started for that folder.
- At most `MAX_PARALLEL_FILTERS` (e.g., 4) run concurrently to balance speed vs. resource usage (configured in `filter_config.py`).

## Shorts/Subtitles Exclusion (yt-dlp)
- Shorts filtered by `--match-filter "duration > 180"`.
- Subtitle-related titles are skipped (`subtitles`, `captions`, `cc`, `multilingual`, etc.).
- Subtitle files are not downloaded: `--no-write-subs --no-write-auto-subs`.

## Duplicate Avoidance
- Before download/processing, check if a sanitized speaker folder already exists in either:
  - `ted_clips_new/` (queue) or
  - `filtered_videos_2/` (final output).
- If found, skip the video to avoid duplicates.
