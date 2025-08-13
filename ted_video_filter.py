#!/usr/bin/env python3
"""
TED Talk Video Filter
Filters existing TED talk video collection to ensure each video contains only one speaker without audience.
Removes non-TED talk videos and multi-speaker content.
"""

import os
import sys
import shutil
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import subprocess
import tempfile
from dataclasses import dataclass

# Import existing tools
try:
    from tool import (
        evaluate_window_person_presence, 
        load_yolo_model, 
        ensure_ffmpeg_on_path,
        extract_wav_for_vad,
        read_wav_as_bytes,
        run_vad
    )
except ImportError:
    print("ERROR: Could not import from tool.py. Make sure tool.py is in the same directory.")
    sys.exit(1)

# Import configuration
try:
    from filter_config import *
    INPUT_DIR = Path(INPUT_DIR)
    OUTPUT_DIR = Path(OUTPUT_DIR)
    TEMP_DIR = Path(TEMP_DIR)
except ImportError:
    # Fallback configuration if config file not found
    INPUT_DIR = Path("ted_clips_400")
    OUTPUT_DIR = Path("filtered_ted_clips")
    TEMP_DIR = Path("temp_filter")
    MIN_SINGLE_PERSON_RATIO = 0.7
    MAX_AUDIENCE_RATIO = 0.2
    MIN_CLIP_DURATION = 10
    MAX_CLIPS_PER_SPEAKER = 3
    NON_TED_KEYWORDS = [
        "fake", "parody", "reaction", "review", "commentary", "compilation",
        "best of", "worst", "funny", "fails", "meme", "joke", "satire",
        "browns", "oasis", "live", "concert", "music", "song", "cover",
        "game", "translator", "philharmonic", "kyle kulinski", "josh talks hindi"
    ]

@dataclass
class ClipAnalysis:
    """Analysis results for a video clip"""
    file_path: Path
    duration: float
    single_person_ratio: float
    avg_persons: float
    max_persons: float
    slide_ratio: float
    is_valid: bool
    reason: str

def log(message: str):
    """Simple logging with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    try:
        print(f"[{timestamp}] {message}")
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(f"[{timestamp}] {safe_message}")

def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", 
            "-show_format", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        log(f"Error getting duration for {video_path}: {e}")
        return 0.0

def is_non_ted_talk(speaker_name: str, video_title: str = "") -> bool:
    """Check if this appears to be a non-TED talk video based on name/title"""
    combined_text = f"{speaker_name} {video_title}".lower()
    
    for keyword in NON_TED_KEYWORDS:
        if keyword in combined_text:
            return True
    
    # Check for suspicious patterns
    if any(pattern in combined_text for pattern in [
        "#", "vs", "x ", " x ", "collab", "feat.", "ft.", 
        "episode", "ep.", "part ", "pt.", "season", "s0", "s1", "s2"
    ]):
        return True
    
    return False

def analyze_clip(clip_path: Path, yolo_model) -> ClipAnalysis:
    """Analyze a single video clip for person presence and content quality"""
    try:
        duration = get_video_duration(clip_path)
        
        if duration < MIN_CLIP_DURATION:
            return ClipAnalysis(
                file_path=clip_path,
                duration=duration,
                single_person_ratio=0.0,
                avg_persons=0.0,
                max_persons=0.0,
                slide_ratio=0.0,
                is_valid=False,
                reason=f"Too short ({duration:.1f}s < {MIN_CLIP_DURATION}s)"
            )
        
        # Pull tunables from config (fall back to sane defaults if missing)
        sample_every = globals().get("SAMPLE_EVERY_SECONDS", 2.0)
        max_samples = int(globals().get("MAX_SAMPLES_PER_CLIP", 24))
        yolo_conf = float(globals().get("YOLO_CONFIDENCE", 0.25))
        min_box_area_ratio = float(globals().get("MIN_BOX_AREA_RATIO", 0.01))
        target_width = int(globals().get("TARGET_FRAME_WIDTH", 320))
        avoid_slides = bool(globals().get("AVOID_SLIDES", True))

        # Analyze person presence throughout the clip
        # Use HOG detector since YOLO has NumPy compatibility issues
        avg_persons, max_persons, single_person_ratio, slide_ratio = evaluate_window_person_presence(
            str(clip_path),
            0.0,  # start_sec
            duration,  # end_sec
            sample_every=sample_every,
            max_samples=max_samples,
            detector="hog",
            yolo_model=None,
            samples_per_window=None,
            target_width=target_width,
            avoid_slides=avoid_slides
        )
        
        # Determine if clip is valid based on criteria
        max_slide_ratio = float(globals().get("MAX_SLIDE_RATIO", 0.3))
        is_valid = (
            single_person_ratio >= MIN_SINGLE_PERSON_RATIO and
            avg_persons <= 1.5 and  # Allow slight detection noise
            slide_ratio <= max_slide_ratio
        )
        
        reason = "Valid single-speaker clip"
        if single_person_ratio < MIN_SINGLE_PERSON_RATIO:
            reason = f"Low single-person ratio ({single_person_ratio:.2f} < {MIN_SINGLE_PERSON_RATIO})"
        elif avg_persons > 1.5:
            reason = f"Too many people on average ({avg_persons:.2f})"
        elif slide_ratio > max_slide_ratio:
            reason = f"Too much slide content ({slide_ratio:.2f})"
        
        return ClipAnalysis(
            file_path=clip_path,
            duration=duration,
            single_person_ratio=single_person_ratio,
            avg_persons=avg_persons,
            max_persons=max_persons,
            slide_ratio=slide_ratio,
            is_valid=is_valid,
            reason=reason
        )
        
    except Exception as e:
        log(f"Error analyzing {clip_path}: {e}")
        return ClipAnalysis(
            file_path=clip_path,
            duration=0.0,
            single_person_ratio=0.0,
            avg_persons=0.0,
            max_persons=0.0,
            slide_ratio=0.0,
            is_valid=False,
            reason=f"Analysis error: {e}"
        )

def process_speaker_folder(speaker_folder: Path, yolo_model) -> List[ClipAnalysis]:
    """Process all clips in a speaker folder and return analysis results"""
    log(f"Processing speaker: {speaker_folder.name}")
    
    # Check if this appears to be a non-TED talk
    if is_non_ted_talk(speaker_folder.name):
        log(f"Skipping non-TED talk: {speaker_folder.name}")
        return []
    
    all_clips = []
    
    # Find all video clips in subfolders
    for subfolder in speaker_folder.iterdir():
        if subfolder.is_dir():
            for clip_file in subfolder.glob("*.mp4"):
                all_clips.append(clip_file)
    
    if not all_clips:
        log(f"No video clips found in {speaker_folder.name}")
        return []
    
    log(f"Found {len(all_clips)} clips for {speaker_folder.name}")
    
    # Analyze each clip
    analyses = []
    for clip_path in all_clips:
        analysis = analyze_clip(clip_path, yolo_model)
        analyses.append(analysis)
        
        status = "✓" if analysis.is_valid else "✗"
        log(f"  {status} {clip_path.name}: {analysis.reason}")
    
    return analyses

def copy_filtered_clips(analyses: List[ClipAnalysis], speaker_name: str) -> int:
    """Copy valid clips to the filtered output directory"""
    valid_analyses = [a for a in analyses if a.is_valid]
    
    if not valid_analyses:
        return 0
    
    # Sort by single_person_ratio (best first) and take top clips
    valid_analyses.sort(key=lambda x: x.single_person_ratio, reverse=True)
    max_keep = int(globals().get("MAX_CLIPS_PER_SPEAKER", 1))
    if max_keep < 1:
        max_keep = 1
    selected_clips = valid_analyses[:max_keep]
    
    # Create output directory for this speaker
    speaker_output_dir = OUTPUT_DIR / speaker_name
    speaker_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy selected clips
    copied_count = 0
    for i, analysis in enumerate(selected_clips, 1):
        output_filename = f"clip_{i:02d}.mp4"
        output_path = speaker_output_dir / output_filename
        
        try:
            shutil.copy2(analysis.file_path, output_path)
            copied_count += 1
            log(f"  Copied: {output_filename} (ratio: {analysis.single_person_ratio:.2f})")
        except Exception as e:
            log(f"  Error copying {analysis.file_path}: {e}")
    
    return copied_count

def generate_filter_report(all_analyses: Dict[str, List[ClipAnalysis]], total_copied: int):
    """Generate a detailed filtering report"""
    report_path = OUTPUT_DIR / "filter_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("TED Talk Video Filtering Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        total_speakers = len(all_analyses)
        total_clips_analyzed = sum(len(analyses) for analyses in all_analyses.values())
        valid_clips = sum(len([a for a in analyses if a.is_valid]) for analyses in all_analyses.values())
        
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total speakers processed: {total_speakers}\n")
        f.write(f"Total clips analyzed: {total_clips_analyzed}\n")
        f.write(f"Valid single-speaker clips: {valid_clips}\n")
        f.write(f"Clips copied to filtered folder: {total_copied}\n")
        f.write(f"Success rate: {(valid_clips/total_clips_analyzed*100):.1f}%\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 20 + "\n")
        
        for speaker_name, analyses in all_analyses.items():
            if not analyses:  # Skipped non-TED talks
                f.write(f"\n{speaker_name}: SKIPPED (Non-TED talk)\n")
                continue
                
            valid_count = len([a for a in analyses if a.is_valid])
            f.write(f"\n{speaker_name}: {valid_count}/{len(analyses)} valid clips\n")
            
            for analysis in analyses:
                status = "✓" if analysis.is_valid else "✗"
                f.write(f"  {status} {analysis.file_path.name}: {analysis.reason}\n")
                if analysis.is_valid:
                    f.write(f"    Single-person ratio: {analysis.single_person_ratio:.2f}\n")
                    f.write(f"    Avg persons: {analysis.avg_persons:.2f}\n")
    
    log(f"Detailed report saved to: {report_path}")

def main():
    """Main filtering process"""
    log("Starting TED Talk Video Filtering")
    log(f"Input directory: {INPUT_DIR}")
    log(f"Output directory: {OUTPUT_DIR}")
    
    # Check dependencies
    ensure_ffmpeg_on_path()
    
    if not INPUT_DIR.exists():
        log(f"ERROR: Input directory not found: {INPUT_DIR}")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Load YOLO model for person detection
    log("Loading YOLO model for person detection...")
    try:
        yolo_model = load_yolo_model()
        log("YOLO model loaded successfully")
    except Exception as e:
        log(f"ERROR: Could not load YOLO model: {e}")
        sys.exit(1)
    
    # Process each speaker folder
    all_analyses = {}
    total_copied = 0
    
    speaker_folders = [f for f in INPUT_DIR.iterdir() if f.is_dir()]
    log(f"Found {len(speaker_folders)} speaker folders to process")
    
    for i, speaker_folder in enumerate(speaker_folders, 1):
        log(f"\n[{i}/{len(speaker_folders)}] Processing: {speaker_folder.name}")
        
        try:
            analyses = process_speaker_folder(speaker_folder, yolo_model)
            all_analyses[speaker_folder.name] = analyses
            
            if analyses:  # Only copy if we have valid analyses (not skipped)
                copied = copy_filtered_clips(analyses, speaker_folder.name)
                total_copied += copied
                
                if copied > 0:
                    log(f"✓ Copied {copied} clips for {speaker_folder.name}")
                else:
                    log(f"✗ No valid clips found for {speaker_folder.name}")
            
        except Exception as e:
            log(f"ERROR processing {speaker_folder.name}: {e}")
            all_analyses[speaker_folder.name] = []
    
    # Generate report
    log(f"\nGenerating filtering report...")
    generate_filter_report(all_analyses, total_copied)
    
    # Cleanup
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR)
        except:
            pass
    
    # Final summary
    log(f"\n{'='*50}")
    log(f"FILTERING COMPLETED!")
    log(f"Total clips copied: {total_copied}")
    log(f"Output directory: {OUTPUT_DIR.absolute()}")
    log(f"Detailed report: {OUTPUT_DIR / 'filter_report.txt'}")
    log(f"{'='*50}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nFiltering interrupted by user")
        # Cleanup temp directory
        if TEMP_DIR.exists():
            try:
                shutil.rmtree(TEMP_DIR)
            except:
                pass
    except Exception as e:
        log(f"Unexpected error: {e}")
        raise
