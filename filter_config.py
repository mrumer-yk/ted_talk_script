#!/usr/bin/env python3
"""
Configuration file for TED Talk Video Filter
Adjust these settings to customize the filtering behavior
"""

# Directory Configuration
INPUT_DIR = "ted_clips_new"           # Source directory paths
OUTPUT_DIR = "filtered_videos_2" # Output directory for filtered clips
TEMP_DIR = "temp_filter"              # Temporary directory for processing

# Filtering Criteria
MIN_SINGLE_PERSON_RATIO = 0.1         # At least 10% of frames should have exactly one person (very permissive)
MAX_AUDIENCE_RATIO = 0.8              # Maximum 80% of frames can have multiple people
MIN_CLIP_DURATION = 10                # Minimum clip duration in seconds (shorter clips are rejected)
MAX_CLIPS_PER_SPEAKER = 1             # Keep ONLY ONE best clip per speaker

# Person Detection Settings
YOLO_CONFIDENCE = 0.25                # Lower confidence to catch smaller persons
MIN_BOX_AREA_RATIO = 0.01             # Detect smaller person boxes
SAMPLE_EVERY_SECONDS = 2.0            # Sample frames more frequently
MAX_SAMPLES_PER_CLIP = 24             # Analyze a few more samples per clip
TARGET_FRAME_WIDTH = 320              # Resize frames to this width for faster processing

# Content Quality Filters
MAX_SLIDE_RATIO = 0.3                 # Maximum ratio of slide/presentation content allowed
AVOID_SLIDES = True                   # Whether to actively avoid slide-heavy content

# Non-TED Talk Detection Keywords (case insensitive)
# Videos containing these keywords in speaker name or title will be filtered out
NON_TED_KEYWORDS = [
    # Fake/Parody content
    "fake", "parody", "reaction", "review", "commentary", "compilation",
    "best of", "worst", "funny", "fails", "meme", "joke", "satire",
    
    # Music/Entertainment
    "browns", "oasis", "live", "concert", "music", "song", "cover",
    "philharmonic", "beatbox", "guitar", "performance",
    
    # Gaming/Tech
    "game", "translator", "tesla", "gigafactory",
    
    # Other non-TED content
    "kyle kulinski", "josh talks hindi", "shorts", "tiktok", "instagram",
    "episode", "season", "part", "pt.", "ep.", "vs", "x ", " x ",
    "collab", "feat.", "ft.", "collaboration"
]

# Suspicious Patterns (will be checked in combination with keywords)
SUSPICIOUS_PATTERNS = [
    "#", "vs", "x ", " x ", "collab", "feat.", "ft.", 
    "episode", "ep.", "part ", "pt.", "season", "s0", "s1", "s2",
    "day ", "week ", "month ", "year ", "2023", "2024", "2025"
]

# Logging Configuration
VERBOSE_LOGGING = True                # Enable detailed logging
LOG_ANALYSIS_DETAILS = True          # Log detailed analysis results for each clip

# Performance Settings
PARALLEL_PROCESSING = False           # Enable parallel processing (experimental)
MAX_WORKERS = 4                       # Number of worker threads if parallel processing is enabled

# Report Settings
GENERATE_DETAILED_REPORT = True       # Generate detailed filtering report
INCLUDE_THUMBNAILS = False            # Include thumbnail images in report (requires additional processing)

# Advanced Filters
STRICT_PERSON_DETECTION = True        # Use strict person detection (more accurate but slower)
FILTER_MULTIPLE_SPEAKERS = True       # Filter out clips with multiple distinct speakers
MIN_FACE_SIZE_RATIO = 0.02           # Minimum face size ratio for speaker detection

print("Filter configuration loaded successfully!")
print(f"Input: {INPUT_DIR} -> Output: {OUTPUT_DIR}")
print(f"Criteria: {MIN_SINGLE_PERSON_RATIO:.0%} single-person, max {MAX_CLIPS_PER_SPEAKER} clips per speaker")
