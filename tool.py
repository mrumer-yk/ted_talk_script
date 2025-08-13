import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# Ensure stdout/stderr can print non-ASCII on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

try:
    import webrtcvad  # type: ignore
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    webrtcvad = None  # type: ignore


@dataclass
class WindowScore:
    start_sec: float
    end_sec: float
    speech_ratio: float


def ensure_ffmpeg_on_path() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("ffmpeg is required but not found on PATH. Please install ffmpeg and retry.", file=sys.stderr)
        sys.exit(1)


def download_youtube(url: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        url,
    ]
    subprocess.run(cmd, check=True)

    # Pick the newest mp4 in directory
    mp4s = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.lower().endswith(".mp4")]
    if not mp4s:
        raise RuntimeError("Download appears to have failed; no MP4 found.")
    newest = max(mp4s, key=os.path.getmtime)
    return newest


def extract_wav_for_vad(video_path: str, tmp_dir: str) -> str:
    wav_path = os.path.join(tmp_dir, "audio_16k_mono.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        wav_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav_path


def read_wav_as_bytes(wav_path: str) -> Tuple[bytes, int]:
    # Minimal WAV reader for PCM16 mono 16k
    with open(wav_path, "rb") as f:
        data = f.read()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Unsupported WAV format")
    # Find fmt and data chunks
    idx = 12
    sample_rate = None
    num_channels = None
    bits_per_sample = None
    pcm_data = None
    while idx + 8 <= len(data):
        chunk_id = data[idx:idx+4]
        chunk_size = int.from_bytes(data[idx+4:idx+8], "little")
        chunk_data = data[idx+8:idx+8+chunk_size]
        if chunk_id == b"fmt ":
            audio_format = int.from_bytes(chunk_data[0:2], "little")
            num_channels = int.from_bytes(chunk_data[2:4], "little")
            sample_rate = int.from_bytes(chunk_data[4:8], "little")
            bits_per_sample = int.from_bytes(chunk_data[14:16], "little")
            if audio_format != 1:
                raise ValueError("Only PCM WAV supported")
        elif chunk_id == b"data":
            pcm_data = chunk_data
        idx += 8 + chunk_size
        if idx % 2 == 1:
            idx += 1
    if sample_rate != 16000 or num_channels != 1 or bits_per_sample != 16 or pcm_data is None:
        raise ValueError("Expected mono PCM16 16k WAV")
    return pcm_data, sample_rate  # type: ignore


# --- VAD paths ---

def run_vad_webrtc(pcm16: bytes, sample_rate: int, frame_ms: int, vad_aggr: int) -> np.ndarray:
    if not WEBRTCVAD_AVAILABLE or webrtcvad is None:
        return run_vad_fallback(pcm16, sample_rate, frame_ms)
    vad = webrtcvad.Vad(vad_aggr)
    frame_bytes = int(sample_rate * (frame_ms / 1000.0) * 2)
    num_frames = len(pcm16) // frame_bytes
    flags = np.zeros(num_frames, dtype=np.uint8)
    for i in range(num_frames):
        frame = pcm16[i*frame_bytes:(i+1)*frame_bytes]
        try:
            flags[i] = 1 if vad.is_speech(frame, sample_rate) else 0
        except Exception:
            flags[i] = 0
    return flags


def run_vad_fallback(pcm16: bytes, sample_rate: int, frame_ms: int) -> np.ndarray:
    # Pure numpy heuristic VAD
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    frame_len = int(sample_rate * (frame_ms / 1000.0))
    num_frames = len(samples) // frame_len

    # Truncate to full frames
    samples = samples[: num_frames * frame_len]
    frames = samples.reshape(num_frames, frame_len)

    # Energy per frame
    energy = (frames ** 2).mean(axis=1)

    # Zero-crossing rate per frame
    signs = np.sign(frames)
    signs[signs == 0] = 1
    zcr = (np.diff(signs, axis=1) != 0).mean(axis=1)

    # Spectral band energy ratio (300-3400 Hz over 0-8000 Hz)
    # rfft size: next power of 2
    nfft = 1 << (frame_len - 1).bit_length()
    window = np.hanning(frame_len).astype(np.float32)
    spec_mag = np.abs(np.fft.rfft(frames * window, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    voice_band = (freqs >= 300) & (freqs <= 3400)
    total_energy = (spec_mag ** 2).sum(axis=1) + 1e-9
    voice_energy = (spec_mag[:, voice_band] ** 2).sum(axis=1)
    voice_ratio = voice_energy / total_energy

    # Heuristics: speech tends to have moderate ZCR, sufficient energy, and high voice-band ratio
    # Thresholds tuned for 16k mono
    energy_thr = max(1e-4, np.median(energy) * 1.5)
    zcr_low, zcr_high = 0.02, 0.20
    voice_ratio_thr = max(0.4, np.median(voice_ratio))

    speech = (
        (energy > energy_thr)
        & (zcr > zcr_low)
        & (zcr < zcr_high)
        & (voice_ratio > voice_ratio_thr)
    )
    return speech.astype(np.uint8)


def run_vad(pcm16: bytes, sample_rate: int, frame_ms: int, vad_aggr: int) -> np.ndarray:
    if WEBRTCVAD_AVAILABLE:
        return run_vad_webrtc(pcm16, sample_rate, frame_ms, vad_aggr)
    return run_vad_fallback(pcm16, sample_rate, frame_ms)


# --- Vision detection backends ---

def load_yolo_model() -> Optional[object]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception:
        return None
    # Prefer local yolov8n.pt if present
    weights = None
    for cand in ["yolov8n.pt", os.path.join(os.getcwd(), "yolov8n.pt")]:
        if os.path.exists(cand):
            weights = cand
            break
    try:
        model = YOLO(weights or "yolov8n.pt")
        return model
    except Exception:
        return None


def detect_persons_yolo(frame: np.ndarray, model: object, conf: float = 0.4, min_box_area_ratio: float = 0.015) -> int:
    h, w = frame.shape[:2]
    area_thresh = max(1.0, min_box_area_ratio * float(h * w))
    try:
        results = model.predict(source=frame, imgsz=640, conf=conf, verbose=False)
    except Exception:
        return 0
    count = 0
    for r in results or []:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        try:
            cls = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            for i, c in enumerate(cls):
                if int(c) != 0:  # person class is 0, skip non-person classes
                    continue
                x1, y1, x2, y2 = xyxy[i]
                box_area = max(0.0, (x2 - x1) * (y2 - y1))
                if box_area >= area_thresh:
                    count += 1
        except Exception:
            continue
    return count


def count_persons_in_frame_hog(frame: np.ndarray) -> int:
    import cv2  # local import so script runs without OpenCV when not needed

    hog = getattr(count_persons_in_frame_hog, "_hog", None)
    if hog is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        count_persons_in_frame_hog._hog = hog  # type: ignore[attr-defined]
    # Resize to speed up processing
    target_w = 320
    h, w = frame.shape[:2]
    scale = target_w / float(max(w, 1))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    rects, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return 0 if rects is None else len(rects)


def is_slide_frame(frame: np.ndarray) -> bool:
    """Heuristic slide detection: looks for a large axis-aligned rectangle area (typical slide)
    and high edge density when no person is detected. Fast and works well for full-screen slides.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return False

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)

    # Edge density as a quick early signal
    edge_ratio = float(np.count_nonzero(edges)) / float(h * w)

    # Find large rectangular regions (slides)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.25 * h * w:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        # Bounding rect geometry checks
        x, y, bw, bh = cv2.boundingRect(approx)
        if bw < 0.5 * w or bh < 0.5 * h:
            continue
        aspect = bw / float(bh + 1e-6)
        if 1.2 <= aspect <= 2.0:  # ~4:3 to 16:9
            # Slide likely present
            return True

    # If edges are very dense and no big rectangle found, still avoid
    return edge_ratio > 0.20


# --- Vision window evaluation ---

def vision_score_window(video_path: str, start_sec: float, end_sec: float, sample_every: float = 2.0, max_samples: int = 40) -> Tuple[float, float]:
    """Returns (avg_persons, max_persons) across sampled frames in the window.
    If OpenCV is unavailable, returns (np.inf, np.inf) to signal unusable vision score.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        return (float("inf"), float("inf"))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (float("inf"), float("inf"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    persons_counts: List[int] = []
    duration = max(0.0, end_sec - start_sec)
    num_samples = max(1, min(int(math.ceil(duration / max(sample_every, 1e-3))), max_samples))

    for i in range(num_samples):
        t = start_sec + min(duration, i * sample_every)
        ts_msec = max(0.0, t * 1000.0)
        ok_seek = False
        try:
            ok_seek = bool(cap.set(cv2.CAP_PROP_POS_MSEC, ts_msec))
        except Exception:
            ok_seek = False
        if not ok_seek:
            try:
                frame_idx = int(max(0, t * fps))
                ok_seek = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx))
            except Exception:
                ok_seek = False
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        try:
            persons = count_persons_in_frame_hog(frame)
            persons_counts.append(persons)
        except Exception:
            continue

    cap.release()

    if not persons_counts:
        return (float("inf"), float("inf"))

    avg_persons = float(np.mean(persons_counts))
    max_persons = float(np.max(persons_counts))
    return (avg_persons, max_persons)


def evaluate_window_person_presence(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sample_every: float,
    max_samples: int,
    detector: str = "auto",
    yolo_model: Optional[object] = None,
    yolo_conf: float = 0.4,
    min_box_area_ratio: float = 0.015,
    samples_per_window: Optional[int] = None,
    target_width: int = 320,
    avoid_slides: bool = False,
) -> Tuple[float, float, float, float]:
    """Returns (avg_persons, max_persons, ratio_single_person) across sampled frames.
    If detector backend unavailable, returns (inf, inf, 0.0).
    """
    try:
        import cv2  # noqa: F401
    except Exception:
        return (float("inf"), float("inf"), 0.0, 1.0)

    # Resolve backend
    backend = detector
    model = yolo_model
    if backend == "auto":
        model = model or load_yolo_model()
        backend = "yolo" if model is not None else "hog"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (float("inf"), float("inf"), 0.0, 1.0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    persons_counts: List[int] = []
    single_flags: List[bool] = []
    slide_flags: List[bool] = []
    duration = max(0.0, end_sec - start_sec)
    if samples_per_window is not None and samples_per_window > 0:
        num_samples = samples_per_window
        times = np.linspace(start_sec, end_sec, num=num_samples, endpoint=False)
    else:
        num_samples = max(1, min(int(math.ceil(duration / max(sample_every, 1e-3))), max_samples))
        times = [start_sec + min(duration, i * sample_every) for i in range(num_samples)]

    for t in times:
        ts_msec = max(0.0, t * 1000.0)
        ok_seek = False
        try:
            ok_seek = bool(cap.set(cv2.CAP_PROP_POS_MSEC, ts_msec))
        except Exception:
            ok_seek = False
        if not ok_seek:
            try:
                frame_idx = int(max(0, t * fps))
                ok_seek = bool(cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx))
            except Exception:
                ok_seek = False
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        try:
            # Downscale for speed
            h, w = frame.shape[:2]
            if w > target_width:
                scale = target_width / float(w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            if backend == "yolo" and model is not None:
                count = detect_persons_yolo(frame, model, conf=yolo_conf, min_box_area_ratio=min_box_area_ratio)
            else:
                count = count_persons_in_frame_hog(frame)
            persons_counts.append(count)
            single_flags.append(count == 1)
            if avoid_slides:
                slide_flags.append(is_slide_frame(frame))
        except Exception:
            continue

    cap.release()

    if not persons_counts:
        return (float("inf"), float("inf"), 0.0, 1.0)

    avg_persons = float(np.mean(persons_counts))
    max_persons = float(np.max(persons_counts))
    ratio_single = float(np.mean(single_flags)) if single_flags else 0.0
    slide_ratio = float(np.mean(slide_flags)) if slide_flags else 0.0
    return (avg_persons, max_persons, ratio_single, slide_ratio)


def combine_speech_and_vision(scores: List[WindowScore], video_path: str, sample_every: float, topk: int = 50) -> List[WindowScore]:
    # Evaluate at most topk windows by speech to keep it fast
    subset = scores[:max(1, min(topk, len(scores)))]

    scored: List[Tuple[WindowScore, float]] = []
    for s in tqdm(subset, desc="Vision re-rank", unit="win"):
        avg_p, max_p = vision_score_window(video_path, s.start_sec, s.end_sec, sample_every=sample_every)
        # If vision failed, keep a neutral penalty so audio still drives the decision
        if not np.isfinite(avg_p):
            vision_weight = 1.0
        else:
            # Prefer exactly one person; penalize >1 strongly, <0.5 slightly (rare detection misses)
            if avg_p <= 1.2 and max_p <= 2.0:
                vision_weight = 1.0
            elif avg_p <= 2.0 and max_p <= 3.0:
                vision_weight = 0.6
            else:
                vision_weight = 0.25
        combined = s.speech_ratio * vision_weight
        scored.append((s, combined))

    # Re-rank by combined score and return merged with remaining (audio-only) windows
    scored.sort(key=lambda t: t[1], reverse=True)
    reranked = [t[0] for t in scored]
    if len(reranked) < len(scores):
        reranked.extend(scores[len(reranked):])
    return reranked


# --- Scoring and selection ---

def score_windows(
    vad_flags: np.ndarray,
    frame_ms: int,
    clip_duration: int,
    stride_seconds: float,
    edge_margin: float,
    max_seconds: Optional[float] = None,
) -> List[WindowScore]:
    frames_per_sec = int(1000 / frame_ms)
    window_frames = clip_duration * frames_per_sec
    stride_frames = max(1, int(stride_seconds * frames_per_sec))

    # Ignore margins
    start_idx = int(edge_margin * frames_per_sec)
    end_idx = max(0, len(vad_flags) - int(edge_margin * frames_per_sec))
    if max_seconds is not None and max_seconds > 0:
        end_limit = int(max_seconds * frames_per_sec)
        end_idx = min(end_idx, end_limit)

    scores: List[WindowScore] = []
    i = start_idx
    while i + window_frames <= end_idx:
        window = vad_flags[i:i+window_frames]
        speech_ratio = float(window.sum()) / float(window_frames)
        start_sec = i / frames_per_sec
        end_sec = (i + window_frames) / frames_per_sec
        scores.append(WindowScore(start_sec, end_sec, speech_ratio))
        i += stride_frames
    scores.sort(key=lambda s: s.speech_ratio, reverse=True)
    return scores


def pick_non_overlapping(scores: List[WindowScore], num: int, min_gap: float = 0.0) -> List[WindowScore]:
    selected: List[WindowScore] = []
    for s in scores:
        overlaps = False
        for t in selected:
            if not (s.end_sec <= t.start_sec - min_gap or s.start_sec >= t.end_sec + min_gap):
                overlaps = True
                break
        if not overlaps:
            selected.append(s)
            if len(selected) == num:
                break
    selected.sort(key=lambda s: s.start_sec)
    return selected


def cut_clip_ffmpeg(video_path: str, out_path: str, start_sec: float, duration_sec: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        video_path,
        "-t",
        str(duration_sec),
        "-c",
        "copy",
        out_path,
    ]
    # Some MP4s require re-encoding for clean segmenting; if copy fails, re-encode
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            video_path,
            "-t",
            str(duration_sec),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            out_path,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def get_speaker_from_title(title: str) -> str:
    parts = title.split('|')
    if len(parts) >= 2:
        # Typically 'Title | Speaker | TED', so speaker is the second part
        speaker = parts[1].strip()
        # Avoid generic names like 'TED'
        if speaker.lower() != 'ted':
            return speaker
    # Fallback to the original title if speaker not found
    return title

def sanitize_name(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in (" ", "_", "-"))
    return safe.strip().replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a talk video and extract speech-heavy 30s clips.")
    parser.add_argument("url", nargs="?", default=None, help="YouTube URL (optional if --input-file provided)")
    parser.add_argument("--input-file", default=None, help="Path to a local video file to process instead of downloading")
    parser.add_argument("--output-dir", default="out", help="Directory to store downloads and clips")
    parser.add_argument("--num-clips", type=int, default=5, help="Number of clips to export")
    parser.add_argument("--clip-duration", type=int, default=30, help="Clip duration in seconds")
    parser.add_argument("--vad-aggr", type=int, default=2, choices=[0, 1, 2, 3], help="WebRTC VAD aggressiveness")
    parser.add_argument("--stride-seconds", type=float, default=5.0, help="Stride for sliding window in seconds")
    parser.add_argument("--frame-ms", type=int, default=30, choices=[10, 20, 30], help="VAD frame size in ms")
    parser.add_argument("--edge-margin", type=float, default=10.0, help="Ignore this many seconds at start/end")

    # Vision options
    parser.add_argument("--vision-filter", action="store_true", help="Prefer clips with only one person visible (OpenCV HOG)")
    parser.add_argument("--vision-sample-seconds", type=float, default=2.0, help="Sample interval for vision scoring (sec)")
    parser.add_argument("--vision-topk", type=int, default=50, help="Max candidate windows to vision-score")

    # Advanced detector options similar to the reference tool
    parser.add_argument("--detector", choices=["auto", "hog", "yolo"], default="auto", help="Vision backend for person detection")
    parser.add_argument("--strict-person-only", action="store_true", help="Require segments to be single-person in most sampled frames")
    parser.add_argument("--strict-min-ratio", type=float, default=0.7, help="Min ratio of sampled frames that must have exactly one person")
    parser.add_argument("--min-box-area-ratio", type=float, default=0.015, help="Min bbox area ratio to count as a person (ignore small audience)")
    parser.add_argument("--yolo-conf", type=float, default=0.4, help="YOLO confidence threshold")
    # Slide filtering
    parser.add_argument("--avoid-slides", action="store_true", help="Avoid frames that look like slides/presentations")
    parser.add_argument("--slide-max-ratio", type=float, default=0.2, help="Max allowed slide frame ratio per window")

    args = parser.parse_args()

    ensure_ffmpeg_on_path()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Input file not found: {args.input_file}", file=sys.stderr)
            sys.exit(1)
        video_path = args.input_file
        print(f"Using local file: {video_path}")
    else:
        if not args.url:
            print("Provide a YouTube URL or --input-file.", file=sys.stderr)
            sys.exit(1)
        print("Downloading video...")
        video_path = download_youtube(args.url, args.output_dir)
        print(f"Downloaded: {video_path}")

    with tempfile.TemporaryDirectory() as tmp:
        print("Extracting audio for VAD...")
        wav_path = extract_wav_for_vad(video_path, tmp)
        pcm16, sr = read_wav_as_bytes(wav_path)
        if WEBRTCVAD_AVAILABLE:
            print("Running WebRTC VAD...")
        else:
            print("webrtcvad not available; using lightweight fallback VAD.")
        vad_flags = run_vad(pcm16, sr, args.frame_ms, args.vad_aggr)

    print("Scoring windows (audio)...")
    scores = score_windows(
        vad_flags=vad_flags,
        frame_ms=args.frame_ms,
        clip_duration=args.clip_duration,
        stride_seconds=args.stride_seconds,
        edge_margin=args.edge_margin,
    )

    if not scores:
        print("No valid windows found. Try lowering edge margin or using a shorter clip duration.", file=sys.stderr)
        sys.exit(2)

    # Advanced strict person-only filter
    if args.strict_person_only:
        print("Applying strict person-only filter...")
        yolo_model = load_yolo_model() if args.detector in ("auto", "yolo") else None
        # Limit number of windows considered to speed up
        candidate_windows = scores[:min(len(scores), max(10, min(30, len(scores))))]
        filtered: List[WindowScore] = []
        for s in tqdm(candidate_windows, desc="Strict person-only", unit="win"):
            avg_p, max_p, ratio_single, slide_ratio = evaluate_window_person_presence(
                video_path,
                s.start_sec,
                s.end_sec,
                sample_every=max(2.5, args.vision_sample_seconds),
                max_samples=20,
                detector=args.detector,
                yolo_model=yolo_model,
                yolo_conf=args.yolo_conf,
                min_box_area_ratio=args.min_box_area_ratio,
                samples_per_window=3,
                target_width=320,
                avoid_slides=args.avoid_slides,
            )
            if ratio_single >= args.strict_min_ratio and (not args.avoid_slides or slide_ratio <= args.slide_max_ratio):
                filtered.append(s)
        # If nothing passes strictly, fall back to top speech windows
        scores = filtered if filtered else scores

    # Legacy simple re-ranking vision filter
    if (not args.strict_person_only) and args.vision_filter:
        print("Applying vision filter (preferring single-person shots)...")
        scores = combine_speech_and_vision(scores, video_path, sample_every=args.vision_sample_seconds, topk=args.vision_topk)

    selected = pick_non_overlapping(scores, args.num_clips)
    if not selected:
        print("No non-overlapping windows found.", file=sys.stderr)
        sys.exit(3)

    # Extract speaker name from video title to use as folder name
    video_title = os.path.splitext(os.path.basename(video_path))[0]
    speaker_name = get_speaker_from_title(video_title)
    sane_speaker_name = sanitize_name(speaker_name)

    clip_dir = os.path.join(args.output_dir, sane_speaker_name)
    os.makedirs(clip_dir, exist_ok=True)

    print("Cutting clips...")
    for idx, s in enumerate(selected, start=1):
        out_path = os.path.join(clip_dir, f"clip_{idx:02d}.mp4")
        cut_clip_ffmpeg(video_path, out_path, s.start_sec, args.clip_duration)
        print(f"Saved {out_path} (speech_ratio={s.speech_ratio:.2f})")

    print("Done.")
    print(f"Clips saved in: {os.path.abspath(clip_dir)}")


if __name__ == "__main__":
    main()
