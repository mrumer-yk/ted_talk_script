#!/usr/bin/env python3
"""
TED Talk Video Filter Runner
Simple script to run the filtering process with progress tracking
"""

import sys
import time
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run TED Talk video filtering")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-continue without confirmation prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("TED TALK VIDEO FILTER")
    print("=" * 60)
    print()
    
    # Check if required files exist
    required_files = ["ted_video_filter.py", "tool.py", "filter_config.py"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ ERROR: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print()
        print("Please ensure all required files are in the current directory.")
        return 1
    
    # Check if input directory exists
    try:
        from filter_config import INPUT_DIR
        input_path = Path(INPUT_DIR)
    except ImportError:
        input_path = Path("ted_clips_400")
    
    if not input_path.exists():
        print(f"❌ ERROR: Input directory not found: {input_path}")
        print("Please ensure your TED talk videos are in the correct directory.")
        return 1
    
    # Show configuration summary
    try:
        from filter_config import (
            OUTPUT_DIR, MIN_SINGLE_PERSON_RATIO, MAX_CLIPS_PER_SPEAKER,
            MIN_CLIP_DURATION, NON_TED_KEYWORDS
        )
        
        print("📋 CONFIGURATION SUMMARY:")
        print(f"   Input Directory: {INPUT_DIR}")
        print(f"   Output Directory: {OUTPUT_DIR}")
        print(f"   Single-person threshold: {MIN_SINGLE_PERSON_RATIO:.0%}")
        print(f"   Max clips per speaker: {MAX_CLIPS_PER_SPEAKER}")
        print(f"   Min clip duration: {MIN_CLIP_DURATION}s")
        print(f"   Non-TED keywords: {len(NON_TED_KEYWORDS)} patterns")
        print()
        
    except ImportError:
        print("⚠️  Using default configuration (filter_config.py not found)")
        print()
    
    # Count input videos
    speaker_folders = [f for f in input_path.iterdir() if f.is_dir()]
    total_clips = 0
    for folder in speaker_folders:
        for subfolder in folder.iterdir():
            if subfolder.is_dir():
                total_clips += len(list(subfolder.glob("*.mp4")))
    
    print(f"📊 INPUT ANALYSIS:")
    print(f"   Speaker folders: {len(speaker_folders)}")
    print(f"   Total video clips: {total_clips}")
    print()
    
    # Confirm before starting
    print("🚀 Ready to start filtering process!")
    print("This may take a while depending on the number of videos...")
    print()
    
    if not args.yes:
        if sys.stdin and hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            response = input("Do you want to continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Filtering cancelled.")
                return 0
        else:
            print("Non-interactive session detected: continuing without prompt.")
    else:
        print("--yes provided: continuing without prompt.")
    
    print()
    print("Starting filtering process...")
    print("-" * 60)
    
    # Import and run the main filter
    try:
        from ted_video_filter import main as filter_main
        start_time = time.time()
        filter_main()
        end_time = time.time()
        
        print()
        print("=" * 60)
        print("✅ FILTERING COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {(end_time - start_time)/60:.1f} minutes")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Filtering interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR during filtering: {e}")
        print("Check the error details above for troubleshooting.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
