# Automated TED Talk Video Curation

This repository presents a high-level overview of a system that automatically curates single-speaker video segments from public talks.

The pipeline runs continuously and performs:

1. Discovery
   - Periodically lists recent videos from a public talk channel.
   - Skips shorts using a duration gate and ignores titles that indicate hardcoded subtitles.

2. Candidate generation
   - Downloads a talk and extracts audio to locate speech-dense windows using voice activity detection (VAD).
   - Ranks fixed-length windows by speech presence and selects non-overlapping candidates.

3. Visual verification
   - Samples frames within each candidate window and estimates how many people are visible using a lightweight detector.
   - Prefers windows with exactly one visible person and avoids slide-heavy frames via fast heuristics.

4. Parallel second-stage filtering
   - As soon as a speakers candidates are generated, a second process validates them with stricter thresholds.
   - The best clip is promoted to a final collection if it meets quality criteria.

5. De-duplication and safety checks
   - Skips speakers already present in either the candidate queue or the final collection.
   - Avoids duplicate processing across runs.

Key design choices:
- Audio-first, vision-second ranking to keep the system fast and robust without heavyweight models.
- Simple, explainable heuristics for slide avoidance and single-person preference.
- Parallelism in the second stage to increase throughput while keeping resource use predictable.

Outputs:
- Short, speech-dense segments with a single visible speaker, suitable for downstream tasks and demos.

Notes:
- Requires a standard multimedia toolkit and common Python packages (for audio/vision and video I/O).
- No source files are included in this public view; this is a process summary intended for interviews.

Sample outputs:
- A small gallery of filtered clips is available for review.
