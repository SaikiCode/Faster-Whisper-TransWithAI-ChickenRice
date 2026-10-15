import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faster_whisper_transwithai_chickenrice.injection as injection
from faster_whisper_transwithai_chickenrice.vad_manager import VadConfig


class SharedManagerTests(unittest.TestCase):
    def tearDown(self) -> None:
        injection._global_manager = None

    def test_lazily_created_manager_is_reused_across_calls(self) -> None:
        injection._global_manager = None
        with mock.patch.object(injection, "VadModelManager") as manager_cls:
            manager_cls.return_value.get_speech_timestamps.return_value = []
            for _ in range(3):
                injection.get_speech_timestamps_injected(np.zeros(16_000, dtype=np.float32))

        manager_cls.assert_called_once()
        self.assertEqual(manager_cls.return_value.get_speech_timestamps.call_count, 3)

    def test_injected_manager_is_used_without_creating_another(self) -> None:
        existing = mock.Mock()
        existing.get_speech_timestamps.return_value = [{"start": 0, "end": 320}]
        with (
            mock.patch.object(injection, "VadModelManager") as manager_cls,
            mock.patch.object(injection.mock, "patch"),
        ):
            injection.inject_vad(model_id="whisper_vad", config=VadConfig(), manager=existing)
            try:
                result = injection.get_speech_timestamps_injected(np.zeros(16_000, dtype=np.float32))
            finally:
                injection.uninject_vad()

        manager_cls.assert_not_called()
        self.assertEqual(result, [{"start": 0, "end": 320}])
        self.assertIsNone(injection._global_manager)


class PrecomputedProbsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = mock.Mock(frame_duration_ms=20)
        self.manager.get_speech_timestamps.return_value = []
        injection._global_manager = self.manager

    def tearDown(self) -> None:
        injection._global_manager = None

    def passed_probs(self):
        return self.manager.get_speech_timestamps.call_args.kwargs.get("speech_probs")

    def test_matching_probs_are_forwarded(self) -> None:
        probs = np.full(50, 0.9)
        with injection.precomputed_speech_probs(probs):
            injection.get_speech_timestamps_injected(np.zeros(16_000, dtype=np.float32))  # 1s = 50 frames
        self.assertIs(self.passed_probs(), probs)

    def test_mismatched_probs_fall_back_to_running_the_model(self) -> None:
        with injection.precomputed_speech_probs(np.full(80, 0.9)):
            injection.get_speech_timestamps_injected(np.zeros(16_000, dtype=np.float32))
        self.assertIsNone(self.passed_probs())

    def test_context_is_cleared_on_exit(self) -> None:
        with injection.precomputed_speech_probs(np.full(50, 0.9)):
            pass
        injection.get_speech_timestamps_injected(np.zeros(16_000, dtype=np.float32))
        self.assertIsNone(self.passed_probs())


if __name__ == "__main__":
    unittest.main()
