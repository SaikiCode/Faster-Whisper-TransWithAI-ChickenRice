import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np

sys.modules.setdefault("pyjson5", types.SimpleNamespace(decode_io=json.load))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import faster_whisper_transwithai_chickenrice.infer as infer_module
import faster_whisper_transwithai_chickenrice.injection as injection_module
from faster_whisper_transwithai_chickenrice.infer import (
    Inference,
    Segment,
    enforce_segment_timeline,
    vad_segments_to_clip_timestamps,
)


def make_args(config_path: Path, task: str | None = None) -> Namespace:
    return Namespace(
        generation_config=str(config_path),
        task=task,
        vad_threshold=None,
        vad_min_speech_duration_ms=None,
        vad_min_silence_duration_ms=None,
        vad_speech_pad_ms=None,
        merge_segments=None,
        merge_max_gap_ms=None,
        merge_max_duration_ms=None,
        smart_split_with_vad=None,
        target_chunk_duration_s=None,
    )


class WhisperTaskConfigTests(unittest.TestCase):
    def load_config(self, config_text: str, task: str | None = None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "generation_config.json5"
            config_path.write_text(config_text, encoding="utf-8")

            inference = Inference.__new__(Inference)
            config, _segment_merge_options, _smart_split_options = inference._load_generation_config(
                make_args(config_path, task=task)
            )
            return config

    def test_uses_task_from_generation_config(self) -> None:
        config = self.load_config('{"task": "transcribe"}')

        self.assertEqual(config["task"], "transcribe")

    def test_cli_task_overrides_generation_config(self) -> None:
        config = self.load_config('{"task": "transcribe"}', task="translate")

        self.assertEqual(config["task"], "translate")

    def test_cli_task_takes_precedence_over_invalid_config_task(self) -> None:
        config = self.load_config('{"task": "caption"}', task="transcribe")

        self.assertEqual(config["task"], "transcribe")

    def test_invalid_config_task_fails_without_cli_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid Whisper task"):
            self.load_config('{"task": "caption"}')

    def test_defaults_to_translate_without_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "missing.json5"
            inference = Inference.__new__(Inference)
            config, _segment_merge_options, _smart_split_options = inference._load_generation_config(
                make_args(config_path)
            )

        self.assertEqual(config["task"], "translate")


class DeviceResolutionTests(unittest.TestCase):
    def test_amd_aliases_map_to_cuda(self) -> None:
        for alias in ("amd", "ROCm", " hip "):
            self.assertEqual(infer_module.resolve_device(alias), "cuda")

    def test_auto_uses_cpu_when_cuda_hidden(self) -> None:
        with mock.patch.dict(infer_module.os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            self.assertEqual(infer_module.resolve_device("auto"), "cpu")

    def test_auto_detects_cuda_devices(self) -> None:
        fake_ct2 = mock.Mock()
        fake_ct2.get_cuda_device_count.return_value = 1
        with (
            mock.patch.dict(infer_module.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}),
            mock.patch.object(infer_module, "ctranslate2", fake_ct2),
        ):
            self.assertEqual(infer_module.resolve_device("auto"), "cuda")

    def test_default_cpu_threads_is_half_of_logical_cores(self) -> None:
        with mock.patch.object(infer_module.os, "cpu_count", return_value=16):
            self.assertEqual(infer_module.default_cpu_threads(), 8)
        with mock.patch.object(infer_module.os, "cpu_count", return_value=1):
            self.assertEqual(infer_module.default_cpu_threads(), 1)


class VadClipTimestampTests(unittest.TestCase):
    def test_converts_sample_indices_to_clip_timestamp_pairs(self) -> None:
        clips = vad_segments_to_clip_timestamps([{"start": 16_000, "end": 32_000}])

        self.assertEqual(clips, [1.0, 2.0])

    def test_converts_sample_indices_to_batched_clip_dicts(self) -> None:
        clips = vad_segments_to_clip_timestamps([{"start": 16_000, "end": 32_000}], batched=True)

        self.assertEqual(clips, [{"start": 1.0, "end": 2.0}])

    def test_prepare_transcription_uses_vad_clips_when_smart_split_disabled(self) -> None:
        inference = Inference.__new__(Inference)
        inference.generation_config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5},
        }
        inference.smart_split_options = infer_module.SmartSplitOptions(enabled=False)
        inference.vad_manager = mock.Mock()
        inference.vad_manager.get_speech_timestamps.return_value = [{"start": 16_000, "end": 32_000}]

        with mock.patch.object(infer_module, "decode_audio", return_value=[0] * 32_000):
            audio, config, duration_after_vad = inference._prepare_transcription("audio.mp3", batched=False)

        self.assertEqual(len(audio), 32_000)
        self.assertEqual(duration_after_vad, 1.0)
        self.assertFalse(config["vad_filter"])
        self.assertEqual(config["clip_timestamps"], [1.0, 2.0])
        self.assertEqual(config["beam_size"], 1)
        self.assertFalse(config["condition_on_previous_text"])


class SmartSplitTests(unittest.TestCase):
    def test_enforce_segment_timeline_caps_and_removes_overlaps(self) -> None:
        segments = [
            Segment(start=1_000, end=30_000, text="first"),
            Segment(start=2_000, end=4_000, text="overlap"),
            Segment(start=5_000, end=5_000, text="empty"),
        ]

        normalized = enforce_segment_timeline(segments, max_duration_ms=20_000)

        self.assertEqual(normalized, [Segment(start=1_000, end=21_000, text="first")])

    def test_smart_chunk_transcription_runs_internal_vad_per_chunk(self) -> None:
        inference = Inference.__new__(Inference)
        inference.generation_config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5},
        }
        inference.smart_split_options = infer_module.SmartSplitOptions(enabled=True, target_chunk_duration_s=2.0)
        # 5s of audio (250 frames): speech 0-1s, pause 1-3s, speech 3-4s, pause 4-5s.
        probs = np.full(250, 0.05, dtype=np.float32)
        probs[0:50] = 0.9
        probs[150:200] = 0.9
        inference.vad_manager = mock.Mock(frame_duration_ms=20)
        inference.vad_manager.get_speech_probs.return_value = probs
        inference.vad_manager.get_speech_timestamps.return_value = [
            {"start": 0, "end": 16_000},
            {"start": 48_000, "end": 64_000},
        ]

        class FakeModel:
            def __init__(self) -> None:
                self.calls = []
                self.probs_seen = []

            def transcribe(self, audio, **config):
                self.calls.append((len(audio), dict(config)))
                self.probs_seen.append(injection_module._precomputed_speech_probs)
                index = len(self.calls)
                segment = mock.Mock(start=0.0, end=0.5, text=f"chunk {index}")
                return [segment], mock.Mock(duration=len(audio) / 16_000, duration_after_vad=0.5)

        model = FakeModel()
        task = mock.Mock(audio_path="audio.mp3")

        with mock.patch.object(infer_module, "decode_audio", return_value=np.zeros(80_000, dtype=np.float32)):
            segments, info = inference._transcribe_smart_chunks(model, task)

        # Chunks are cut in the middle of the in-zone pauses: [0, 1.6s) [1.6, 2.88s) [2.88, 4.48s) [4.48, 5s).
        # The 2nd and 4th chunks contain no outer VAD speech and are skipped.
        self.assertEqual([length for length, _config in model.calls], [25_600, 25_600])
        self.assertTrue(all(call_config["vad_filter"] for _length, call_config in model.calls))
        self.assertNotIn("clip_timestamps", model.calls[0][1])
        self.assertEqual(segments, [Segment(0, 500, "chunk 1"), Segment(2_880, 3_380, "chunk 2")])
        self.assertEqual(info.duration, 5.0)
        self.assertEqual(info.duration_after_vad, 1.0)

        # VAD ran once for the whole file; each chunk got the matching slice of the probabilities.
        inference.vad_manager.get_speech_probs.assert_called_once()
        self.assertIs(inference.vad_manager.get_speech_timestamps.call_args.kwargs["speech_probs"], probs)
        self.assertTrue(np.array_equal(model.probs_seen[0], probs[0:80]))
        self.assertTrue(np.array_equal(model.probs_seen[1], probs[144:224]))
        self.assertIsNone(injection_module._precomputed_speech_probs)

    def test_forced_cut_restarts_next_chunk_after_last_complete_segment(self) -> None:
        inference = Inference.__new__(Inference)
        inference.generation_config = {"language": "ja", "vad_filter": True, "vad_parameters": {"threshold": 0.5}}
        inference.smart_split_options = infer_module.SmartSplitOptions(enabled=True)
        # 40s of continuous speech with a single breath at 26s: no pause, so the first cut is forced at 26s.
        probs = np.full(2_000, 0.9, dtype=np.float32)
        probs[1_300] = 0.5
        inference.vad_manager = mock.Mock(frame_duration_ms=20)
        inference.vad_manager.get_speech_probs.return_value = probs
        inference.vad_manager.get_speech_timestamps.return_value = [{"start": 0, "end": 640_000}]

        def seg(start: float, end: float, text: str):
            return mock.Mock(start=start, end=end, text=text)

        responses = [
            # Chunk [0, 26s): the last sentence is cut off at the chunk end.
            [seg(0.0, 10.0, "a"), seg(10.0, 20.0, "b"), seg(20.5, 27.0, "c (cut)")],
            # Chunk [20s, 40s): restarted after "b", so "c" is decoded whole.
            [seg(0.5, 7.0, "c"), seg(8.0, 19.0, "d")],
        ]
        calls = []

        class FakeModel:
            def transcribe(self, audio, **config):
                calls.append(len(audio))
                return responses[len(calls) - 1], mock.Mock(duration_after_vad=len(audio) / 16_000)

        with mock.patch.object(infer_module, "decode_audio", return_value=np.zeros(640_000, dtype=np.float32)):
            segments, _info = inference._transcribe_smart_chunks(FakeModel(), mock.Mock(audio_path="audio.mp3"))

        self.assertEqual(calls, [26 * 16_000, 20 * 16_000])
        self.assertEqual(
            segments,
            [
                Segment(0, 10_000, "a"),
                Segment(10_000, 20_000, "b"),
                Segment(20_500, 27_000, "c"),
                Segment(28_000, 39_000, "d"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
