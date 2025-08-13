# Local Usage

Follow these steps to run the TED Talk filtering pipeline locally.

## 1) Clone the Repository
```bash
git clone https://github.com/mrumer-yk/ted_talk_script.git
cd ted_talk_script
```

## 2) Install FFmpeg
- Install FFmpeg and ensure `ffmpeg` and `ffprobe` are available on PATH.

## 3) Install Python Dependencies
```bash
pip install -r requirements.txt
```

## 4) Configure Paths and Thresholds
- Open `filter_config.py` and review:
  - Input/Output directories (defaults: `ted_clips_new/`, `filtered_videos_2/`).
  - HOG thresholds: `MIN_SINGLE_PERSON_RATIO`, `MAX_AUDIENCE_RATIO`.
  - Parallelism: `MAX_PARALLEL_FILTERS`.

## 5) Run the Full Pipeline
```bash
python batch_ted_clipper.py
```
What happens:
- Fetches videos from official `@TED` using yt-dlp flat playlist.
- Excludes Shorts (< 180s) and subtitle-likely titles.
- Skips already-processed speakers (checks `ted_clips_new/` and `filtered_videos_2/`).
- Downloads video, clips into 5×30s segments, evaluates using HOG, and saves the best one per speaker.

## 6) Optional: Run Filtering Only
- If you already have clips under `ted_clips_new/Some_Speaker/`:
```bash
python process_single_speaker.py --speaker_dir ted_clips_new/Some_Speaker
```
- Or process all existing speakers in `ted_clips_new/`:
```bash
python run_filter.py
```

## Notes
- This repo intentionally excludes videos, zips, model weights, and temporary outputs (see `.gitignore`).
- Ensure a stable internet connection for yt-dlp.
- If you want stricter/looser filtering, adjust thresholds in `filter_config.py`.
