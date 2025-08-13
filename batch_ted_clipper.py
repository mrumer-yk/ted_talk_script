#!/usr/bin/env python3
"""
Standalone TED Talk Batch Clipper
Downloads top 400 TED Talk videos, creates clips, organizes by unique speakers.
"""

import os
import sys
import json
import subprocess
import shutil
import time
from pathlib import Path
from typing import Set, Dict, List, Optional
import re

# Configuration
OUTPUT_DIR = Path("ted_clips_new")
TEMP_DIR = Path("temp_downloads")
MAX_FILTERED_SPEAKERS = 2000
CLIPS_PER_VIDEO = 5
CLIP_DURATION = 30
VIDEO_QUALITY = "best[height<=1080]"  # High quality but manageable

# Parallel filtering configuration
MAX_PARALLEL_FILTERS = 4  # limit concurrent second-stage filters

# Import filter output directory to monitor progress
try:
    from filter_config import OUTPUT_DIR as FILTERED_OUTPUT_DIR_STR
except Exception:
    FILTERED_OUTPUT_DIR_STR = "filtered_videos_2"
FILTERED_OUTPUT_DIR = Path(FILTERED_OUTPUT_DIR_STR)

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

def log(message: str):
    """Simple logging with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    # Handle Unicode encoding issues on Windows
    try:
        print(f"[{timestamp}] {message}")
    except UnicodeEncodeError:
        # Fallback: replace problematic characters
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(f"[{timestamp}] {safe_message}")

def get_filtered_speaker_count() -> int:
    """Count how many speakers have been finalized in filtered output."""
    if not FILTERED_OUTPUT_DIR.exists():
        return 0
    return len([p for p in FILTERED_OUTPUT_DIR.iterdir() if p.is_dir()])

def sanitize_name(name: str) -> str:
    """Sanitize speaker name for folder creation"""
    # Remove invalid characters for folder names
    safe = re.sub(r'[<>:"/\\|?*]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:50]  # Limit length

def get_speaker_from_title(title: str) -> Optional[str]:
    """Extract speaker name from TED Talk title"""
    # TED Talk titles often follow patterns like:
    # "Title | Speaker Name | TED"
    # "Speaker Name: Title | TED"
    
    if '|' in title:
        parts = [p.strip() for p in title.split('|')]
        # Look for the speaker part (usually not "TED" and not too long)
        for part in parts:
            if part.lower() != 'ted' and len(part) < 50 and len(part) > 3:
                # Check if it looks like a person's name (has spaces, proper case)
                if ' ' in part and part.replace(' ', '').replace('-', '').isalpha():
                    return part
    
    # Fallback: look for pattern "Name: Title"
    if ':' in title:
        potential_speaker = title.split(':')[0].strip()
        if len(potential_speaker) < 50 and ' ' in potential_speaker:
            return potential_speaker
    
    # Last resort: use first part before any separator
    first_part = title.split('|')[0].split(':')[0].strip()
    if len(first_part) < 50:
        return first_part
    
    return None

def fetch_video_urls(limit: int = 1000) -> List[Dict[str, str]]:
    """Fetch video metadata from official TED channel using flat playlist; filter out shorts."""
    log(f"Fetching top {limit} videos from TED channel (@TED)...")
    channel_videos_url = "https://www.youtube.com/@TED/videos"
    try:
        # Use flat playlist to quickly list items without resolving each video
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp",
             "--dump-json",
             "--flat-playlist",
             "--playlist-end", str(limit),
             channel_videos_url],
            capture_output=True, text=True, check=True, encoding='utf-8', timeout=180
        )
        entries = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Expect entries of type 'url' with 'id' and 'title'
                vid = data.get('id')
                title = data.get('title', '')
                if not vid or not title:
                    continue
                entries.append({'id': vid, 'title': title})
            except json.JSONDecodeError:
                continue
        # Filter out likely shorts by approximate title markers later; we will also enforce duration filter at download time
        log(f"Found {len(entries)} entries from @TED. Applying duration filter via yt-dlp per video.")
        return entries
    except subprocess.TimeoutExpired:
        log("Channel listing timed out; using fallback search...")
        return fetch_video_urls_simple(limit // 2)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log(f"Error listing channel videos: {e}")
        return fetch_video_urls_simple(limit // 2)

def fetch_video_urls_simple(limit: int = 500) -> List[Dict[str, str]]:
    """Fallback: Simple TED search without channel restrictions."""
    log(f"Using fallback search for {limit} TED videos...")
    try:
        search_query = f"ytsearch{limit}:TED talk"
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp",
             "--get-title", "--get-id",
             search_query],
            capture_output=True, text=True, check=True, encoding='utf-8', timeout=120
        )
        lines = result.stdout.strip().split('\n')
        videos = []
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                title = lines[i].strip()
                video_id = lines[i + 1].strip()
                if title and video_id:
                    videos.append({'id': video_id, 'title': title})
        log(f"Found {len(videos)} videos using fallback method.")
        return videos
    except Exception as e:
        log(f"Fallback search failed: {e}")
        return []

def download_video(video_id: str, download_folder: Path) -> Optional[Path]:
    """Download a single video by its ID, avoiding hardcoded subtitles."""
    output_template = str(download_folder / f"{video_id}.%(ext)s")
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", VIDEO_QUALITY,
        "-o", output_template,
        "--no-playlist",
        "--match-filter", "duration > 180",
        "--no-write-subs",
        "--no-write-auto-subs"
    ]
    
    try:
        log(f"Downloading video {video_id}...")
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Find the downloaded file
        for file_path in TEMP_DIR.glob(f"{video_id}.*"):
            if file_path.suffix in ['.mp4', '.mkv', '.webm']:
                return file_path
        
        log(f"Warning: Downloaded file for {video_id} not found")
        return None
        
    except subprocess.CalledProcessError as e:
        log(f"Error downloading {video_id}: {e}")
        return None

def create_clips(video_path: Path, speaker_folder: Path) -> bool:
    """Create clips from video using the existing tool.py"""
    if not video_path.exists():
        log(f"Video file not found: {video_path}")
        return False
    
    # Create speaker folder
    speaker_folder.mkdir(exist_ok=True)
    
    # Use YOLO detector as requested by the user
    cmd = [
        sys.executable, "tool.py",
        "--input-file", str(video_path),
        "--num-clips", str(CLIPS_PER_VIDEO),
        "--clip-duration", str(CLIP_DURATION),
        "--output-dir", str(speaker_folder),
        "--detector", "yolo",
        "--strict-person-only",
        "--avoid-slides"
    ]
    
    try:
        log(f"Creating clips in {speaker_folder.name}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if clips were created
        clip_files = list(speaker_folder.glob("**/*.mp4"))
        if clip_files:
            log(f"Created {len(clip_files)} clips for {speaker_folder.name}")
            return True
        else:
            log(f"No clips created for {speaker_folder.name}")
            return False
            
    except subprocess.CalledProcessError as e:
        log(f"Error creating clips: {e}")
        if e.stdout:
            log(f"stdout: {e.stdout}")
        if e.stderr:
            log(f"stderr: {e.stderr}")
        return False

def cleanup_temp_video(video_path: Path):
    """Delete the temporary video file"""
    try:
        if video_path.exists():
            video_path.unlink()
            log(f"Deleted temporary video: {video_path.name}")
    except Exception as e:
        log(f"Error deleting {video_path}: {e}")

def main():
    """Main processing loop"""
    log("Starting TED Talk Batch Clipper")
    log(f"Target (filtered stage): {MAX_FILTERED_SPEAKERS} unique speakers")
    log(f"Clips output directory: {OUTPUT_DIR}")
    log(f"Filtered output directory: {FILTERED_OUTPUT_DIR}")
    
    # Check if tool.py exists
    if not Path("tool.py").exists():
        log("ERROR: tool.py not found in current directory")
        log("Please ensure tool.py is in the same directory as this script")
        return
    
    # Track processed speakers (from clips dir) and already-filtered speakers
    processed_speakers: Set[str] = set()
    filtered_speakers: Set[str] = set()
    # Track running per-speaker filter subprocesses
    running_filters: List[subprocess.Popen] = []
    
    # Check existing folders to avoid reprocessing
    if OUTPUT_DIR.exists():
        existing_folders = [f.name for f in OUTPUT_DIR.iterdir() if f.is_dir()]
        processed_speakers.update(existing_folders)
        log(f"Found {len(processed_speakers)} existing speaker folders")
    # Load already-filtered speakers to avoid re-downloading/processing duplicates
    if FILTERED_OUTPUT_DIR.exists():
        filtered_speakers.update([f.name for f in FILTERED_OUTPUT_DIR.iterdir() if f.is_dir()])
        log(f"Found {len(filtered_speakers)} speakers already in filtered output")
    
    # Fetch video list
    videos = fetch_video_urls(limit=1000)  # Get more to account for filtering
    if not videos:
        log("ERROR: No videos found")
        return
    
    successful_clips = len(processed_speakers)
    
    for i, video in enumerate(videos):
        # Maintain parallel filter slots
        # Clean up finished filter processes
        running_filters = [p for p in running_filters if p.poll() is None]

        # Check filtered progress
        filtered_count = get_filtered_speaker_count()
        log(f"Progress: {filtered_count}/{MAX_FILTERED_SPEAKERS} filtered speakers")
        if filtered_count >= MAX_FILTERED_SPEAKERS:
            log(f"Reached filtered target of {MAX_FILTERED_SPEAKERS} speakers!")
            break
            
        video_id = video.get('id')
        title = video.get('title', '')
        if not video_id or not title:
            continue

        # Filter out videos that likely have hardcoded subtitles
        subtitle_keywords = ['subtitles', 'captions', 'cc:', 'multilingual']
        if any(keyword in title.lower() for keyword in subtitle_keywords):
            log(f"Skipping video with potential subtitles: {title}")
            continue
        
        log(f"\nProcessing video {i+1}/{len(videos)}: {title}")
        
        # Extract speaker name
        speaker = get_speaker_from_title(title)
        if not speaker:
            log(f"Could not extract speaker from title: {title}")
            continue
        
        speaker_safe = sanitize_name(speaker)
        
        # Skip if we already processed this speaker or it's already in final filtered output
        if speaker_safe in processed_speakers or speaker_safe in filtered_speakers:
            where = "clips/queue" if speaker_safe in processed_speakers else "filtered output"
            log(f"Speaker {speaker} already present in {where}, skipping...")
            continue
        
        # Create speaker folder path
        speaker_folder = OUTPUT_DIR / speaker_safe
        
        # Download video
        video_path = download_video(video_id, TEMP_DIR)
        if not video_path:
            log(f"Failed to download video {video_id}")
            continue
        
        # Create clips
        success = create_clips(video_path, speaker_folder)
        
        # Always cleanup the downloaded video
        cleanup_temp_video(video_path)
        
        if success:
            processed_speakers.add(speaker_safe)
            successful_clips += 1
            log(f"[SUCCESS] Created clips for {speaker} -> {speaker_folder}")

            # Spawn second-stage filtering for this speaker when slot available
            # Wait until under parallel limit
            while True:
                # Refresh and prune finished
                running_filters = [p for p in running_filters if p.poll() is None]
                if len(running_filters) < MAX_PARALLEL_FILTERS:
                    break
                time.sleep(0.5)

            try:
                log(f"Launching filter for speaker: {speaker_folder.name}")
                p = subprocess.Popen([
                    sys.executable,
                    "process_single_speaker.py",
                    "--speaker-dir", str(speaker_folder)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                running_filters.append(p)
            except Exception as e:
                log(f"Failed to launch per-speaker filter: {e}")
        else:
            log(f"[FAILED] Failed to create clips for {speaker}")
            # Remove empty folder if created
            if speaker_folder.exists() and not any(speaker_folder.iterdir()):
                speaker_folder.rmdir()
        
        # Small delay to be respectful and allow filters to progress
        time.sleep(1)
    
    # Final summary
    log(f"\n[COMPLETED!]")
    # Final check and wait for filters if still under target
    running_filters = [p for p in running_filters if p.poll() is None]
    if running_filters:
        log(f"Waiting for {len(running_filters)} filter job(s) to finish...")
        for p in running_filters:
            try:
                p.wait(timeout=60)
            except Exception:
                pass

    log(f"Successfully created clips for {successful_clips} speakers")
    log(f"Clips directory: {OUTPUT_DIR}")
    log(f"Filtered directory: {FILTERED_OUTPUT_DIR}")
    
    # Cleanup temp directory
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            log("Cleaned up temporary directory")
    except Exception as e:
        log(f"Warning: Could not clean up temp directory: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nScript interrupted by user")
        # Cleanup any remaining temp files
        try:
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
        except:
            pass
    except Exception as e:
        log(f"Unexpected error: {e}")
        raise
