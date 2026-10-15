"""
Smart split helpers: choose chunk boundaries from frame-level VAD probabilities.

Whisper only sees 30s at a time, so long audio has to be cut somewhere. To keep speech intact the
cut should fall inside a pause. Working on the (smoothed) probabilities instead of the padded VAD
segments lets short pauses qualify and gives a sensible fallback when there is no pause at all.
"""

from dataclasses import dataclass

import numpy as np

# 5 frames = 100ms at the whisper_vad frame rate of 20ms.
PAUSE_SMOOTHING_FRAMES = 5

CHUNK_FINAL = "final"  # chunk reaches the end of the audio
CHUNK_PAUSE = "pause"  # chunk ends inside a pause
CHUNK_FORCED = "forced"  # no pause found; cut at the least speech-like point


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start: float
    end: float
    kind: str = CHUNK_FINAL

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end - self.start)


def smooth_probs(speech_probs: np.ndarray, frames: int = PAUSE_SMOOTHING_FRAMES) -> np.ndarray:
    """Centered moving average; edges are padded with their own values so they don't look like pauses."""
    probs = np.asarray(speech_probs, dtype=np.float32)
    if frames <= 1 or len(probs) == 0:
        return probs
    padded = np.pad(probs, (frames // 2, frames - 1 - frames // 2), mode="edge")
    return np.convolve(padded, np.ones(frames, dtype=np.float32) / frames, mode="valid")


def _longest_pause_midpoint(smoothed: np.ndarray, lo: int, hi: int, pause_threshold: float) -> int | None:
    """Midpoint of the longest run of frames in [lo, hi) below pause_threshold (latest run wins ties)."""
    if hi <= lo:
        return None
    is_pause = (smoothed[lo:hi] < pause_threshold).astype(np.int8)
    edges = np.diff(np.concatenate(([0], is_pause, [0])))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1)
    if len(run_starts) == 0:
        return None
    lengths = run_ends - run_starts
    best = len(lengths) - 1 - int(np.argmax(lengths[::-1]))
    return lo + int((run_starts[best] + run_ends[best]) // 2)


def find_chunk_end(
    smoothed: np.ndarray,
    start_frame: int,
    *,
    max_frames: int,
    split_window_factor: float,
    pause_threshold: float,
    min_search_frames: int,
) -> tuple[int, str]:
    """
    Pick the (exclusive) end frame of the chunk starting at start_frame.

    1. If the rest of the audio fits, take all of it.
    2. Otherwise cut in the longest pause inside the decision zone (the last split_window_factor of
       the allowed length). Only the part of a pause inside the zone counts, so pauses that straddle
       the zone start still qualify.
    3. Otherwise cut in the longest pause anywhere after min_search_frames.
    4. Otherwise cut at the least speech-like frame of the decision zone (a forced cut).
    """
    total_frames = len(smoothed)
    if total_frames - start_frame <= max_frames:
        return total_frames, CHUNK_FINAL

    window_end = start_frame + max_frames
    zone_start = start_frame + max(1, int(max_frames * (1 - split_window_factor)))
    search_start = start_frame + max(1, min(min_search_frames, zone_start - start_frame))

    for lo in (zone_start, search_start):
        cut = _longest_pause_midpoint(smoothed, lo, window_end, pause_threshold)
        if cut is not None:
            return cut, CHUNK_PAUSE

    return zone_start + _middle_of_lowest(smoothed[zone_start:window_end]), CHUNK_FORCED


def _middle_of_lowest(values: np.ndarray) -> int:
    """Index of the lowest value; smoothing turns a short dip into a flat valley, so take its middle."""
    lowest = np.flatnonzero(values <= values.min() + 1e-6)
    return int(lowest[len(lowest) // 2])


def choose_anchor(
    segment_ends_s: list[float],
    chunk_start_s: float,
    cut_s: float,
    smoothed: np.ndarray,
    frame_s: float,
    *,
    guard_s: float = 0.3,
    min_advance_s: float = 5.0,
    snap_s: float = 0.3,
) -> tuple[int, float] | None:
    """
    After a forced cut, find where the next chunk should start so the cut-off speech is decoded whole.

    Mirrors Whisper's own seek: restart right after the last subtitle segment that ended well before
    the cut (at least guard_s), snapped to the least speech-like frame within snap_s of its end.
    The anchor must be at least min_advance_s into the chunk so processing always moves forward.

    Returns (anchor_frame, keep_until_s): keep the chunk's segments ending at or before keep_until_s
    and start the next chunk at anchor_frame; or None if no segment qualifies.
    """
    earliest = chunk_start_s + min_advance_s
    latest = cut_s - guard_s
    candidates = [end for end in segment_ends_s if earliest <= end <= latest]
    if not candidates:
        return None

    keep_until_s = max(candidates)
    lo = max(int(np.ceil(earliest / frame_s)), int(round((keep_until_s - snap_s) / frame_s)))
    hi = min(int(latest / frame_s), int(round((keep_until_s + snap_s) / frame_s)), len(smoothed) - 1)
    if hi < lo:
        return int(round(keep_until_s / frame_s)), keep_until_s
    return lo + _middle_of_lowest(smoothed[lo : hi + 1]), keep_until_s
