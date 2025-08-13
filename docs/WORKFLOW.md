# Pipeline Workflow and Interconnections

This document shows how the components fit together, which file runs first, how parallel filtering works, and how temporary files are cleaned up.

## High-Level Workflow

```text
[User]
  |
  | 1) Start orchestrator
  v
batch_ted_clipper.py  (entry point for full auto mode)
  |
  |-- fetch_video_urls()  -> yt-dlp (flat playlist on @TED)
  |-- download_video()    -> downloads MP4 to TEMP_DIR
  |-- create_clips()      -> calls tool.py to cut 5 x 30s clips per video
  |-- cleanup_temp_video() after clipping (per video)
  |
  |-- For each speaker folder created under ted_clips_new/:
  |     - launch up to MAX_PARALLEL_FILTERS subprocesses:
  |         subprocess.Popen([python, process_single_speaker.py, --speaker-dir S])
  |
  |-- Waits for running filters to finish when needed
  |-- Final TEMP_DIR cleanup on exit
  v
filtered_videos_2/ (final single-speaker clips)
```

## Module Interactions

```text
batch_ted_clipper.py
  ├─ uses yt-dlp via subprocess to list/download videos
  ├─ calls tool.py (clip creation via ffmpeg)
  ├─ spawns process_single_speaker.py (parallel workers)
  └─ reads thresholds/paths from filter_config.py (for output monitoring)

process_single_speaker.py
  ├─ imports ted_video_filter.py: process_speaker_folder(), copy_filtered_clips(), log()
  ├─ imports tool.py: ensure_ffmpeg_on_path(), load_yolo_model (not used here), etc.
  └─ reads OUTPUT_DIR, TEMP_DIR from filter_config.py

 ted_video_filter.py
  ├─ imports tool.py: evaluate_window_person_presence() (HOG/YOLO), ensure_ffmpeg_on_path()
  ├─ applies HOG person detection (yolo_model=None path) for single-speaker scoring
  ├─ copies best clips to filtered_videos_2/
  └─ writes filter_report.txt inside output folder

 tool.py
  ├─ ffmpeg helpers (audio extraction, clip cutting)
  ├─ VAD helpers (optional fallback)
  ├─ HOG person counter and simple slide heuristic (OpenCV)
  └─ Optional YOLO helpers (ultralytics) — not used in the parallel path

 run_filter.py (optional entry)
  └─ Convenience runner that calls ted_video_filter.main() for existing folders
```

## Which File Runs First?
- Full automation from YouTube: run `batch_ted_clipper.py`.
  - It fetches, downloads, clips, launches parallel filters, and cleans temp files.
- Filtering for existing local clips: run `run_filter.py` (calls `ted_video_filter.main()`), or call `process_single_speaker.py` for one speaker.

## Parallel Processing Model
- In `batch_ted_clipper.py`:
  - `MAX_PARALLEL_FILTERS` controls concurrency (default 4).
  - After creating clips for a speaker in `ted_clips_new/{Speaker}/`, it starts:
    ```bash
    python process_single_speaker.py --speaker-dir ted_clips_new/{Speaker}
    ```
  - The orchestrator keeps a list of running `Popen` processes, prunes finished ones, and only launches new workers if the count is below the limit.
- In `process_single_speaker.py`:
  - Calls `process_speaker_folder()` from `ted_video_filter.py` with `yolo_model=None` to enforce HOG detection.
  - Then calls `copy_filtered_clips()` to write successful clips into `filtered_videos_2/`.

## Temporary Files and Cleanup
- Per-video temporary download:
  - `batch_ted_clipper.download_video()` saves to `TEMP_DIR` (e.g., `temp_downloads/`).
  - After `create_clips()` completes, `cleanup_temp_video()` deletes the downloaded MP4.
- Final orchestrator cleanup:
  - At the end of `batch_ted_clipper.main()`, it removes the entire `TEMP_DIR` if it exists.
- Filtering temp (if any):
  - `ted_video_filter.py` creates its `TEMP_DIR` from `filter_config.py` and cleans it at the end of `main()`.

## Data Flow Summary
```text
YouTube (@TED) → yt-dlp → TEMP_DIR/video.mp4 → tool.py (ffmpeg) → ted_clips_new/{Speaker}/clip_XX.mp4
→ process_single_speaker.py (HOG) → ted_video_filter.process_speaker_folder() → filtered_videos_2/
→ filter_report.txt (summary)
```
