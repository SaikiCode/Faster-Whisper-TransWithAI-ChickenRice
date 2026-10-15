import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faster_whisper_transwithai_chickenrice.smart_split import (
    CHUNK_FINAL,
    CHUNK_FORCED,
    CHUNK_PAUSE,
    choose_anchor,
    find_chunk_end,
    smooth_probs,
)

# 20ms frames: 1500 frames = 30s, decision zone = last 40% = frames 900-1500 of a chunk starting at 0.
FRAMES_PER_S = 50
SPEECH = 0.9
SILENCE = 0.05


def speech(seconds: float) -> np.ndarray:
    return np.full(int(seconds * FRAMES_PER_S), SPEECH, dtype=np.float32)


def silence_at(probs: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    probs = probs.copy()
    probs[int(start_s * FRAMES_PER_S) : int(end_s * FRAMES_PER_S)] = SILENCE
    return probs


def chunk_end(probs: np.ndarray, start_frame: int = 0) -> tuple[int, str]:
    return find_chunk_end(
        smooth_probs(probs),
        start_frame,
        max_frames=30 * FRAMES_PER_S,
        split_window_factor=0.4,
        pause_threshold=0.35,
        min_search_frames=FRAMES_PER_S,
    )


class FindChunkEndTests(unittest.TestCase):
    def test_rest_of_audio_fits_in_one_chunk(self) -> None:
        self.assertEqual(chunk_end(speech(20)), (1000, CHUNK_FINAL))
        self.assertEqual(chunk_end(speech(60), start_frame=1600), (3000, CHUNK_FINAL))

    def test_pause_straddling_zone_start_is_used_instead_of_hard_cut(self) -> None:
        # Pause 15-19.5s starts before the 18s decision zone; the old gap search ignored it and cut at 30s.
        end, kind = chunk_end(silence_at(speech(60), 15.0, 19.5))
        self.assertEqual(kind, CHUNK_PAUSE)
        self.assertGreaterEqual(end, 18 * FRAMES_PER_S)
        self.assertLess(end, int(19.5 * FRAMES_PER_S))

    def test_short_pause_in_zone_beats_forced_cut(self) -> None:
        # 160ms pause at 24s: too short to survive VAD padding as a gap, but still a real pause.
        end, kind = chunk_end(silence_at(speech(60), 24.0, 24.16))
        self.assertEqual(kind, CHUNK_PAUSE)
        self.assertEqual(end, 1204)

    def test_pause_before_zone_is_used_when_zone_has_none(self) -> None:
        end, kind = chunk_end(silence_at(speech(60), 10.0, 11.0))
        self.assertEqual(kind, CHUNK_PAUSE)
        self.assertGreaterEqual(end, 500)
        self.assertLess(end, 550)

    def test_longest_pause_in_zone_wins_and_latest_wins_ties(self) -> None:
        probs = silence_at(silence_at(speech(60), 20.0, 20.5), 25.0, 26.0)
        self.assertEqual(chunk_end(probs)[0] // FRAMES_PER_S, 25)

        probs = silence_at(silence_at(speech(60), 20.0, 21.0), 25.0, 26.0)
        self.assertEqual(chunk_end(probs)[0] // FRAMES_PER_S, 25)

    def test_no_pause_forces_cut_at_least_speech_like_point_in_zone(self) -> None:
        probs = speech(60)
        probs[1300] = 0.5  # a breath: still speech, but the weakest point
        self.assertEqual(chunk_end(probs), (1300, CHUNK_FORCED))

    def test_cut_is_relative_to_chunk_start(self) -> None:
        probs = silence_at(speech(90), 50.0, 51.0)  # chunk starting at 30s, pause at 50s (zone starts at 48s)
        end, kind = chunk_end(probs, start_frame=1500)
        self.assertEqual(kind, CHUNK_PAUSE)
        self.assertGreaterEqual(end, 2500)
        self.assertLess(end, 2550)


class ChooseAnchorTests(unittest.TestCase):
    def anchor(self, ends: list[float], probs: np.ndarray, start_s: float = 0.0, cut_s: float = 30.0):
        return choose_anchor(ends, start_s, cut_s, smooth_probs(probs), 1 / FRAMES_PER_S)

    def test_restarts_after_last_segment_that_ended_before_the_cut(self) -> None:
        probs = speech(60)
        probs[1450] = 0.4  # breath right after the 28.9s segment end
        # 30.0 is the segment clipped at the cut: it is not complete, so it is not a candidate.
        self.assertEqual(self.anchor([7.5, 12.0, 28.9, 30.0], probs), (1450, 28.9))

    def test_anchor_stays_close_to_segment_end_on_flat_speech(self) -> None:
        anchor_frame, keep_until = self.anchor([20.0, 30.0], speech(60))
        self.assertEqual(keep_until, 20.0)
        self.assertEqual(anchor_frame, 1000)

    def test_no_anchor_without_a_complete_segment_far_enough_into_the_chunk(self) -> None:
        self.assertIsNone(self.anchor([2.0, 4.9, 29.8], speech(60)))
        self.assertIsNone(self.anchor([], speech(60)))

    def test_anchor_always_moves_forward(self) -> None:
        anchor_frame, _keep_until = self.anchor([35.1, 59.0], speech(90), start_s=30.0, cut_s=60.0)
        self.assertGreaterEqual(anchor_frame, 35 * FRAMES_PER_S)


class SmoothProbsTests(unittest.TestCase):
    def test_keeps_length_including_inputs_shorter_than_window(self) -> None:
        self.assertEqual(len(smooth_probs(np.full(3, SPEECH))), 3)
        self.assertEqual(len(smooth_probs(speech(10))), 500)
        self.assertEqual(len(smooth_probs(np.zeros(0))), 0)

    def test_edges_are_not_treated_as_pauses(self) -> None:
        smoothed = smooth_probs(speech(1))
        self.assertTrue(np.allclose(smoothed, SPEECH))


if __name__ == "__main__":
    unittest.main()
