import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faster_whisper_transwithai_chickenrice.vad_manager import (
    VadConfig,
    WhisperVadModel,
    get_speech_timestamps_onnx,
    speech_probs_to_timestamps,
)


def make_probs() -> np.ndarray:
    n = 3000
    i = np.arange(n)
    probs = np.full(n, 0.05)
    for start, end in [(50, 400), (408, 700), (760, 1500), (1503, 1520), (1600, 2400), (2410, 2950)]:
        probs[start:end] = 0.9
    probs = probs + 0.08 * np.sin(i * 0.37)
    probs[1000:1004] = 0.3  # short dip inside speech
    probs[2000:2012] = 0.2  # longer dip inside speech
    return np.clip(probs, 0.0, 1.0)


class FakeWrapper:
    frame_duration_ms = 20
    sample_rate = 16_000

    def __init__(self, probs: np.ndarray) -> None:
        self.probs = probs
        self.forward_calls = 0

    def reset_states(self) -> None:
        pass

    def audio_forward(self, audio: np.ndarray, sr: int = 16_000) -> np.ndarray:
        self.forward_calls += 1
        return self.probs


# Recorded from get_speech_timestamps_onnx before it was split into speech_probs_to_timestamps.
GOLDEN_CASES = {
    "project_defaults": (
        dict(threshold=0.5, min_speech_duration_ms=300, min_silence_duration_ms=100, speech_pad_ms=200),
        [
            (12800, 130560, 0.901148),
            (130560, 227200, 0.900478),
            (240000, 489600, 0.893784),
            (508800, 643200, 0.900216),
            (643200, 771200, 0.899639),
            (771200, 947200, 0.900461),
        ],
    ),
    "long_silence": (
        dict(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=2000, speech_pad_ms=400),
        [(9600, 960000, 0.836032)],
    ),
    "explicit_neg": (
        dict(
            threshold=0.6, neg_threshold=0.25, min_speech_duration_ms=100, min_silence_duration_ms=60, speech_pad_ms=30
        ),
        [
            (15680, 128320, 0.901148),
            (130240, 224320, 0.900478),
            (242880, 486720, 0.893784),
            (511680, 640320, 0.900216),
            (643520, 768320, 0.899639),
            (770880, 944320, 0.900461),
        ],
    ),
}


def as_tuples(segments: list[dict[str, float]]) -> list[tuple[int, int, float]]:
    return [(int(s["start"]), int(s["end"]), round(float(s["probability"]), 6)) for s in segments]


class SpeechProbsToTimestampsTests(unittest.TestCase):
    def test_matches_recorded_output(self) -> None:
        probs = make_probs()
        for name, (kwargs, expected) in GOLDEN_CASES.items():
            with self.subTest(name):
                segments = speech_probs_to_timestamps(
                    probs, frame_duration_ms=20, sampling_rate=16_000, return_seconds=False, **kwargs
                )
                self.assertEqual(as_tuples(segments), expected)

    def test_onnx_wrapper_path_matches_probs_path(self) -> None:
        probs = make_probs()
        kwargs = GOLDEN_CASES["project_defaults"][0]
        via_model = get_speech_timestamps_onnx(
            np.zeros(10), FakeWrapper(probs), sampling_rate=16_000, return_seconds=False, **kwargs
        )
        via_probs = speech_probs_to_timestamps(
            probs, frame_duration_ms=20, sampling_rate=16_000, return_seconds=False, **kwargs
        )
        self.assertEqual(as_tuples(via_model), as_tuples(via_probs))


class WhisperVadModelProbsTests(unittest.TestCase):
    def make_model(self, probs: np.ndarray) -> tuple[WhisperVadModel, FakeWrapper]:
        model = WhisperVadModel(VadConfig(onnx_model_path=None))
        wrapper = FakeWrapper(probs)
        model.wrapper = wrapper  # type: ignore[assignment]
        return model, wrapper

    def test_get_speech_probs_trims_window_padding(self) -> None:
        # 3000 frames cover 60s, but the audio is only 10.01s long -> ceil(10.01s / 20ms) = 501 frames.
        model, _wrapper = self.make_model(make_probs())
        probs = model.get_speech_probs(np.zeros(160_160, dtype=np.float32))
        self.assertEqual(len(probs), 501)

    def test_precomputed_probs_skip_the_model(self) -> None:
        model, wrapper = self.make_model(make_probs())
        segments = model.get_speech_timestamps(
            np.zeros(960_000, dtype=np.float32),
            speech_probs=make_probs(),
            **GOLDEN_CASES["project_defaults"][0],
        )
        self.assertEqual(wrapper.forward_calls, 0)
        self.assertEqual(as_tuples(segments), GOLDEN_CASES["project_defaults"][1])

    def test_precomputed_probs_work_without_loaded_model(self) -> None:
        model = WhisperVadModel(VadConfig(onnx_model_path=None))
        segments = model.get_speech_timestamps(
            np.zeros(960_000, dtype=np.float32),
            speech_probs=make_probs(),
            **GOLDEN_CASES["project_defaults"][0],
        )
        self.assertEqual(len(segments), 6)


if __name__ == "__main__":
    unittest.main()
