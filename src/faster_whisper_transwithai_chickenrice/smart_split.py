"""
Smart split helpers: choose chunk boundaries from frame-level VAD probabilities.

Whisper only sees 30s at a time, so long audio has to be cut somewhere. To keep speech intact the
cut should fall inside a pause. Working on the (smoothed) probabilities instead of the padded VAD
segments lets short pauses qualify and gives a sensible fallback when there is no pause at all.
"""

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# 5 frames = 100ms at the whisper_vad frame rate of 20ms.
PAUSE_SMOOTHING_FRAMES = 5

# A batched chunk is decoded again sequentially (with temperature fallback and seek) when more of its
# speech than this is left without any subtitle segment.
MIN_UNCOVERED_SPEECH_S = 1.5
MAX_UNCOVERED_SPEECH_RATIO = 0.2
MIN_SPEECH_FOR_EMPTY_OUTPUT_S = 1.0

# Silence kept between concatenated speech pieces (capped by the real pause) so Whisper still hears
# where a sentence stops and puts a segment boundary there.
MAX_KEPT_GAP_S = 0.5

# Characters after which a subtitle's text may be cut when it is split at a pause.
TEXT_CUT_PUNCTUATION = frozenset("，。！？、；：…,.!?;:")

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


@dataclass(frozen=True)
class SpeechMap:
    """Maps times in concatenated speech audio back to the audio it was cut from (seconds)."""

    concat_starts: tuple[float, ...]
    orig_starts: tuple[float, ...]
    durations: tuple[float, ...]

    @property
    def duration_s(self) -> float:
        """Speech duration, without the silence kept between pieces."""
        return float(sum(self.durations))

    def to_original(self, t: float, *, is_end: bool = False) -> float:
        if not self.durations:
            return 0.0
        index = max(0, bisect_right(self.concat_starts, t) - 1)
        if is_end and index > 0 and t <= self.concat_starts[index]:
            # An end time exactly on a join belongs to the end of the previous piece, not the next one.
            index -= 1
        offset = t - self.concat_starts[index]
        if not is_end and offset > self.durations[index] and index + 1 < len(self.durations):
            # A start inside the kept silence belongs to the speech that follows it.
            return self.orig_starts[index + 1]
        return self.orig_starts[index] + min(max(0.0, offset), self.durations[index])

    def to_speech_time(self, t: float) -> float:
        """Map a time in the concatenated audio to the amount of speech before it (kept silence excluded)."""
        if not self.durations:
            return 0.0
        index = max(0, bisect_right(self.concat_starts, t) - 1)
        offset = min(max(0.0, t - self.concat_starts[index]), self.durations[index])
        return float(sum(self.durations[:index])) + offset


def concat_speech(
    audio: np.ndarray, speech_segments: Sequence[dict[str, Any]], sampling_rate: int, *, max_gap_s: float = 0.0
) -> tuple[np.ndarray, SpeechMap]:
    """
    Concatenate the speech segments (sample indices into audio) like faster_whisper's VAD filter does,
    returning the speech-only audio and the map from its timeline back to audio's timeline.

    With max_gap_s > 0, up to that much silence (never more than the real pause) is kept between
    pieces, so the result is never longer than the audio it was cut from.
    """
    pieces: list[list[int]] = []
    for segment in sorted(speech_segments, key=lambda s: s["start"]):
        start = max(0, int(segment["start"]))
        end = min(len(audio), int(segment["end"]))
        if end <= start:
            continue
        if pieces and start <= pieces[-1][1]:
            pieces[-1][1] = max(pieces[-1][1], end)
        else:
            pieces.append([start, end])

    if not pieces:
        return np.zeros(0, dtype=np.float32), SpeechMap((), (), ())

    max_gap_samples = int(round(max_gap_s * sampling_rate))
    concat_starts: list[float] = []
    parts: list[np.ndarray] = []
    position = 0
    for i, (start, end) in enumerate(pieces):
        gap = min(max_gap_samples, start - pieces[i - 1][1]) if i > 0 else 0
        if gap > 0:
            parts.append(np.zeros(gap, dtype=np.float32))
            position += gap
        concat_starts.append(position / sampling_rate)
        parts.append(audio[start:end])
        position += end - start
    speech = np.concatenate(parts).astype(np.float32, copy=False)
    speech_map = SpeechMap(
        tuple(concat_starts),
        tuple(start / sampling_rate for start, _end in pieces),
        tuple((end - start) / sampling_rate for start, end in pieces),
    )
    return speech, speech_map


def _covered_duration(spans: Sequence[tuple[float, float]], limit: float) -> float:
    covered = 0.0
    reached = 0.0
    for start, end in sorted(spans):
        start, end = max(start, reached, 0.0), min(end, limit)
        if end > start:
            covered += end - start
            reached = end
    return covered


def needs_sequential_retry(
    spans: Sequence[tuple[float, float]],
    compression_ratios: Sequence[float],
    avg_logprobs: Sequence[float],
    speech_duration_s: float,
    *,
    compression_ratio_threshold: float | None,
    log_prob_threshold: float | None,
) -> str | None:
    """
    Why a batched (single-pass) decode of a chunk looks unreliable, or None if it looks fine.

    Batched decoding has no temperature fallback and no seek, so a repetition loop, a low-confidence
    decode or an early stop silently loses speech. spans are the segment times in the chunk's
    speech-only timeline.
    """
    if not spans:
        if speech_duration_s >= MIN_SPEECH_FOR_EMPTY_OUTPUT_S:
            return f"no output for {speech_duration_s:.1f}s of speech"
        return None
    if compression_ratio_threshold is not None:
        worst = max(compression_ratios, default=0.0)
        if worst > compression_ratio_threshold:
            return f"compression_ratio {worst:.2f} > {compression_ratio_threshold} (repetition)"
    if log_prob_threshold is not None:
        lowest = min(avg_logprobs, default=0.0)
        if lowest < log_prob_threshold:
            return f"avg_logprob {lowest:.2f} < {log_prob_threshold}"
    uncovered = speech_duration_s - _covered_duration(spans, speech_duration_s)
    if uncovered > max(MIN_UNCOVERED_SPEECH_S, MAX_UNCOVERED_SPEECH_RATIO * speech_duration_s):
        return f"{uncovered:.1f}s of {speech_duration_s:.1f}s speech not covered"
    return None


def _text_cut_positions(text: str) -> list[int]:
    """Positions where text may be cut: before whitespace or after clause punctuation."""
    positions = set()
    for i, char in enumerate(text):
        if char.isspace():
            positions.add(i)
        elif char in TEXT_CUT_PUNCTUATION:
            positions.add(i + 1)
    return sorted(p for p in positions if 0 < p < len(text))


def _char_count(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _assign_text(text: str, stretches: Sequence[Sequence[float]]) -> list[tuple[int, int, str]]:
    """
    Cut text into (first stretch, last stretch, text) parts, as close as possible to each stretch's
    share of the speech; stretches that cannot get their own text stay joined with the previous one.
    """
    total_speech = sum(st[2] for st in stretches)
    cuts = _text_cut_positions(text)
    parts: list[tuple[int, int, str]] = []
    first = 0
    prev_cut = 0
    speech_before = 0.0
    for k in range(len(stretches) - 1):
        speech_before += stretches[k][2]
        candidates = [p for p in cuts if p > prev_cut and text[prev_cut:p].strip() and text[p:].strip()]
        if not candidates:
            break
        target = len(text) * speech_before / total_speech
        cut = min(candidates, key=lambda p: (abs(p - target), p))
        parts.append((first, k, text[prev_cut:cut].strip()))
        prev_cut = cut
        first = k + 1
    parts.append((first, len(stretches) - 1, text[prev_cut:].strip()))
    return parts


def split_at_pauses(
    start_s: float,
    end_s: float,
    text: str,
    speech_pieces: Sequence[tuple[float, float]],
    *,
    min_pause_s: float,
    min_part_speech_s: float = 0.3,
    max_rate_ratio: float = 2.0,
) -> list[tuple[float, float, str]]:
    """
    Split a subtitle segment that runs across long pauses into one segment per stretch of speech.

    speech_pieces are the sorted VAD speech pieces (seconds) overlapping the segment. Pieces closer
    together than min_pause_s form one stretch; stretches with less than min_part_speech_s of speech
    are dropped (the segment is trimmed instead of split). The text is cut at whitespace or clause
    punctuation, as close as possible to each stretch's share of the speech; stretches that cannot get
    their own text stay joined with the previous one. A part whose text would be spoken more than
    max_rate_ratio times faster than the segment's average (a few words flashed over a short stretch)
    means the cut does not fit the speech: its shortest stretch is dropped and the text cut again.
    A segment that covers no speech is returned as is.
    """
    spans = [(max(s, start_s), min(e, end_s)) for s, e in speech_pieces if min(e, end_s) > max(s, start_s)]
    if not spans:
        return [(start_s, end_s, text)]

    stretches: list[list[float]] = []  # [start, end, speech duration]
    for s, e in spans:
        if stretches and s - stretches[-1][1] < min_pause_s:
            stretches[-1][1] = e
            stretches[-1][2] += e - s
        else:
            stretches.append([s, e, e - s])
    if len(stretches) > 1:
        stretches = [st for st in stretches if st[2] >= min_part_speech_s] or [max(stretches, key=lambda st: st[2])]

    while True:
        parts = _assign_text(text, stretches)
        if len(parts) == 1:
            break
        average_rate = _char_count(text) / sum(st[2] for st in stretches)
        rates = [
            (_char_count(part_text) / sum(st[2] for st in stretches[first : last + 1]), first, last)
            for first, last, part_text in parts
        ]
        worst_rate, first, last = max(rates)
        if worst_rate <= max_rate_ratio * average_rate:
            break
        del stretches[min(range(first, last + 1), key=lambda i: stretches[i][2])]

    return [(stretches[first][0], stretches[last][1], part_text) for first, last, part_text in parts]
