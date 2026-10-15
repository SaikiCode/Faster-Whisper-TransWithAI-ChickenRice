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
    concat_speech,
    find_chunk_end,
    needs_sequential_retry,
    smooth_probs,
    split_at_pauses,
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


class ConcatSpeechTests(unittest.TestCase):
    def test_concatenates_speech_and_maps_times_back(self) -> None:
        audio = np.arange(10 * 16_000, dtype=np.float32)
        speech, speech_map = concat_speech(
            audio, [{"start": 16_000, "end": 32_000}, {"start": 80_000, "end": 112_000}], 16_000
        )
        self.assertEqual(len(speech), 48_000)
        self.assertEqual(speech[0], 16_000)
        self.assertEqual(speech[16_000], 80_000)
        self.assertEqual(speech_map.duration_s, 3.0)
        self.assertEqual(speech_map.to_original(0.5), 1.5)
        self.assertEqual(speech_map.to_original(1.5), 5.5)
        # The join at 1.0s is the end of the first piece or the start of the second.
        self.assertEqual(speech_map.to_original(1.0, is_end=True), 2.0)
        self.assertEqual(speech_map.to_original(1.0), 5.0)
        self.assertEqual(speech_map.to_original(99.0), 7.0)

    def test_merges_overlapping_segments_and_clips_to_audio(self) -> None:
        audio = np.zeros(16_000, dtype=np.float32)
        speech, speech_map = concat_speech(
            audio, [{"start": 8_000, "end": 20_000}, {"start": 0, "end": 4_000}, {"start": 3_000, "end": 6_000}], 16_000
        )
        self.assertEqual(len(speech), 6_000 + 8_000)
        self.assertEqual(speech_map.orig_starts, (0.0, 0.5))

    def test_no_speech(self) -> None:
        speech, speech_map = concat_speech(np.zeros(100, dtype=np.float32), [], 16_000)
        self.assertEqual(len(speech), 0)
        self.assertEqual(speech_map.duration_s, 0.0)

    def test_keeps_short_silence_between_pieces(self) -> None:
        audio = np.arange(10 * 16_000, dtype=np.float32) + 1
        # Speech 1-2s, 5-7s and 7.2-8s: pauses of 3s (0.5s kept) and 0.2s (kept whole).
        speech, speech_map = concat_speech(
            audio,
            [{"start": 16_000, "end": 32_000}, {"start": 80_000, "end": 112_000}, {"start": 115_200, "end": 128_000}],
            16_000,
            max_gap_s=0.5,
        )
        self.assertEqual(len(speech), (1.0 + 0.5 + 2.0 + 0.2 + 0.8) * 16_000)
        self.assertTrue(np.all(speech[16_000:24_000] == 0))
        self.assertEqual(speech[24_000], 80_001)
        self.assertAlmostEqual(speech_map.duration_s, 3.8)
        self.assertEqual(speech_map.concat_starts, (0.0, 1.5, 3.7))
        # A start inside kept silence belongs to the next piece, an end to the previous one.
        self.assertEqual(speech_map.to_original(1.2), 5.0)
        self.assertEqual(speech_map.to_original(1.2, is_end=True), 2.0)
        self.assertEqual(speech_map.to_original(2.0), 5.5)
        self.assertAlmostEqual(speech_map.to_original(3.9), 7.4)
        # Speech time skips the kept silence.
        self.assertEqual(speech_map.to_speech_time(1.2), 1.0)
        self.assertEqual(speech_map.to_speech_time(2.0), 1.5)
        self.assertAlmostEqual(speech_map.to_speech_time(4.5), 3.8)

    def test_kept_silence_never_makes_the_result_longer_than_the_audio(self) -> None:
        audio = np.ones(16_000, dtype=np.float32)
        speech, _speech_map = concat_speech(
            audio, [{"start": 0, "end": 4_000}, {"start": 4_800, "end": 16_000}], 16_000, max_gap_s=0.5
        )
        self.assertEqual(len(speech), 16_000)


class SplitAtPausesTests(unittest.TestCase):
    def split(self, start_s, end_s, text, pieces):
        return split_at_pauses(start_s, end_s, text, pieces, min_pause_s=1.5)

    def test_one_part_per_stretch_of_speech_cut_at_spaces(self) -> None:
        self.assertEqual(
            self.split(0.0, 13.0, "一二三 四五六 七八九", [(0.0, 3.0), (5.0, 8.0), (10.0, 13.0)]),
            [(0.0, 3.0, "一二三"), (5.0, 8.0, "四五六"), (10.0, 13.0, "七八九")],
        )

    def test_cuts_after_clause_punctuation(self) -> None:
        self.assertEqual(
            self.split(0.0, 6.0, "你好？我很好", [(0.0, 2.0), (4.0, 6.0)]),
            [(0.0, 2.0, "你好？"), (4.0, 6.0, "我很好")],
        )

    def test_short_pauses_do_not_split_but_trim_to_speech(self) -> None:
        self.assertEqual(self.split(0.0, 7.0, "a b", [(0.5, 3.0), (3.5, 6.0)]), [(0.5, 6.0, "a b")])

    def test_stretch_with_too_little_speech_is_trimmed_off(self) -> None:
        self.assertEqual(self.split(0.0, 10.0, "a b", [(0.0, 0.2), (5.0, 9.0)]), [(5.0, 9.0, "a b")])

    def test_text_without_cut_positions_is_not_split(self) -> None:
        self.assertEqual(self.split(0.0, 6.0, "一二三四五六", [(0.0, 2.0), (4.0, 6.0)]), [(0.0, 6.0, "一二三四五六")])

    def test_segment_without_speech_is_returned_as_is(self) -> None:
        self.assertEqual(self.split(1.0, 2.0, "a b", []), [(1.0, 2.0, "a b")])


class NeedsSequentialRetryTests(unittest.TestCase):
    def check(self, spans, speech_s, ratios=None, logprobs=None) -> bool:
        return needs_sequential_retry(
            spans,
            ratios if ratios is not None else [1.5] * len(spans),
            logprobs if logprobs is not None else [-0.3] * len(spans),
            speech_s,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

    def test_good_decode_is_kept(self) -> None:
        self.assertFalse(self.check([(0.0, 9.5), (9.8, 20.0)], 20.0))

    def test_repetition_or_low_confidence_triggers_retry(self) -> None:
        self.assertTrue(self.check([(0.0, 20.0)], 20.0, ratios=[3.1]))
        self.assertTrue(self.check([(0.0, 20.0)], 20.0, logprobs=[-1.4]))

    def test_early_stop_triggers_retry(self) -> None:
        self.assertTrue(self.check([(0.0, 12.0)], 20.0))  # 8s of speech never decoded
        self.assertFalse(self.check([(0.0, 3.4)], 4.0))  # 0.6s uncovered is within tolerance

    def test_empty_output_retries_only_for_real_speech(self) -> None:
        self.assertTrue(self.check([], 2.0))
        self.assertFalse(self.check([], 0.5))

    def test_thresholds_can_be_disabled(self) -> None:
        self.assertFalse(
            needs_sequential_retry(
                [(0.0, 5.0)], [9.9], [-9.9], 5.0, compression_ratio_threshold=None, log_prob_threshold=None
            )
        )


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
