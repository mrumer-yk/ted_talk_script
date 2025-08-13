#!/usr/bin/env python3
"""
Process a single speaker folder through the filtering pipeline.
This imports functions from ted_video_filter.py and runs analysis+copy for one folder.
"""

import sys
import argparse
from pathlib import Path

# Import filter modules and config
from filter_config import OUTPUT_DIR as FILTER_OUTPUT_DIR_STR, TEMP_DIR as FILTER_TEMP_DIR
from ted_video_filter import (
    process_speaker_folder,
    copy_filtered_clips,
    log,
)
from tool import load_yolo_model, ensure_ffmpeg_on_path


def main():
    parser = argparse.ArgumentParser(description="Process one speaker folder")
    parser.add_argument("--speaker-dir", required=True, help="Path to the speaker folder containing clips")
    args = parser.parse_args()

    speaker_dir = Path(args.speaker_dir)
    if not speaker_dir.exists() or not speaker_dir.is_dir():
        print(f"ERROR: Speaker folder not found: {speaker_dir}")
        return 1

    # Ensure dependencies
    ensure_ffmpeg_on_path()

    # Use HOG detector for second-stage filtering (more reliable)
    log("Using HOG detector for single-speaker processing...")
    yolo_model = None  # Force HOG detector usage

    # Run analysis
    analyses = process_speaker_folder(speaker_dir, yolo_model)

    # Copy results
    copied = 0
    if analyses:
        copied = copy_filtered_clips(analyses, speaker_dir.name)
        if copied > 0:
            log(f"✓ Copied {copied} clip(s) for {speaker_dir.name}")
        else:
            log(f"✗ No valid clips for {speaker_dir.name}")
    else:
        log(f"Skipped or no clips found for {speaker_dir.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
