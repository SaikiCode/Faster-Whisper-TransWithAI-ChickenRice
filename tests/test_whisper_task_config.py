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
            config, *_options = inference._load_generation_config(make_args(config_path, task=task))
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
            with self.assertLogs(infer_module.logger, "WARNING") as logs:
                config, *_options = inference._load_generation_config(make_args(config_path))

        self.assertEqual(config["task"], "translate")
        # A missing config silently drops its temperature/VAD settings, so say where it was looked for.
        self.assertIn(str(config_path), logs.output[0])

    def test_subtitle_cleanup_options_are_read_and_not_passed_to_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "generation_config.json5"
            config_path.write_text(
                json.dumps(
                    {
                        "pause_split": {"enabled": False, "min_pause_ms": 2000},
                        "hallucination_filter": {"max_chars_per_second": 15, "blocklist": ["谢谢"]},
                    }
                ),
                encoding="utf-8",
            )
            inference = Inference.__new__(Inference)
            config, _merge, _smart, pause_split, hallucination_filter = inference._load_generation_config(
                make_args(config_path)
            )

        self.assertNotIn("pause_split", config)
        self.assertNotIn("hallucination_filter", config)
        self.assertEqual(pause_split, infer_module.PauseSplitOptions(enabled=False, min_pause_ms=2000))
        self.assertEqual(
            hallucination_filter,
            infer_module.HallucinationFilterOptions(max_chars_per_second=15.0, blocklist=("谢谢",)),
        )


class SubtitleCleanupTests(unittest.TestCase):
    def test_segment_across_long_pauses_is_split_per_stretch_of_speech(self) -> None:
        # Speech 169.4-174.0s, 176.2-180.8s and 184.9-187.1s, decoded as one subtitle (the 02:49 case).
        vad_segments = [
            {"start": int(169.4 * 16_000), "end": int(174.0 * 16_000)},
            {"start": int(176.2 * 16_000), "end": int(180.8 * 16_000)},
            {"start": int(184.9 * 16_000), "end": int(187.1 * 16_000)},
        ]
        segments = [
            Segment(160_000, 165_000, "before"),
            Segment(169_440, 187_120, "喂 琉璃 泰尔莉莎 要是打算就这样睡着的话 我就要亲姐姐咯 醒醒 早上好"),
        ]

        result = infer_module.split_segments_at_pauses(segments, vad_segments, infer_module.PauseSplitOptions())

        self.assertEqual(
            result,
            [
                Segment(160_000, 165_000, "before"),
                Segment(169_440, 174_000, "喂 琉璃 泰尔莉莎"),
                Segment(176_200, 180_800, "要是打算就这样睡着的话 我就要亲姐姐咯"),
                Segment(184_900, 187_100, "醒醒 早上好"),
            ],
        )

    def test_pause_split_can_be_disabled(self) -> None:
        segments = [Segment(0, 10_000, "a b")]
        vad_segments = [{"start": 0, "end": 16_000}, {"start": 128_000, "end": 160_000}]

        result = infer_module.split_segments_at_pauses(
            segments, vad_segments, infer_module.PauseSplitOptions(enabled=False)
        )

        self.assertEqual(result, segments)

    def test_hallucination_filter_drops_impossible_speech_rates_and_blocklisted_text(self) -> None:
        segments = [
            Segment(0, 1_000, "我希望你能够在这段时间里 把你的精液尽情地射给我"),  # 22 chars in 1s
            Segment(2_000, 3_000, "感谢您的视听。"),
            Segment(4_000, 5_000, "完"),
            Segment(6_000, 6_500, "嗯 好的"),  # fast but too short to judge
            Segment(7_000, 10_000, "这样啊 那么晚上就可以大家一起亲热了呢"),
        ]

        result = infer_module.filter_hallucinations(segments, infer_module.HallucinationFilterOptions())

        self.assertEqual(result, [segments[3], segments[4]])

    def test_hallucination_filter_can_be_disabled(self) -> None:
        segments = [Segment(0, 100, "感谢您的视听")]

        result = infer_module.filter_hallucinations(segments, infer_module.HallucinationFilterOptions(enabled=False))

        self.assertEqual(result, segments)


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
            audio, config, duration_after_vad = inference._prepare_transcription("audio.mp3")

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


def raw_segment(start: float, end: float, text: str, compression_ratio: float = 1.2, avg_logprob: float = -0.3):
    return mock.Mock(start=start, end=end, text=text, compression_ratio=compression_ratio, avg_logprob=avg_logprob)


def oom_error() -> RuntimeError:
    return RuntimeError("CUDA failed with error out of memory")


class BatchedModeTests(unittest.TestCase):
    def make_inference(self) -> Inference:
        inference = Inference.__new__(Inference)
        inference.generation_config = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
            "vad_parameters": {"threshold": 0.5},
        }
        inference.smart_split_options = infer_module.SmartSplitOptions(enabled=True, target_chunk_duration_s=2.0)
        inference.enable_batching = True
        inference.batch_size = 0

        # Same 5s layout as the sequential test: speech 0-1s and 3-4s, chunks [0, 1.6) and [2.88, 4.48) have speech.
        probs = np.full(250, 0.05, dtype=np.float32)
        probs[0:50] = 0.9
        probs[150:200] = 0.9

        def get_speech_timestamps(model_id, audio, sampling_rate, speech_probs, **kwargs):
            if len(audio) == 80_000:
                return [{"start": 0, "end": 16_000}, {"start": 48_000, "end": 64_000}]
            if speech_probs[0] > 0.5:  # chunk starting at 0s: speech at 0-1s
                return [{"start": 0, "end": 16_000}]
            return [{"start": 1_920, "end": 17_920}]  # chunk starting at 2.88s: speech at 3-4s

        inference.vad_manager = mock.Mock(frame_duration_ms=20)
        inference.vad_manager.get_speech_probs.return_value = probs
        inference.vad_manager.get_speech_timestamps.side_effect = get_speech_timestamps
        return inference

    def run_batched(self, inference: Inference, batched_model, model=None):
        with mock.patch.object(infer_module, "decode_audio", return_value=np.zeros(80_000, dtype=np.float32)):
            return inference._transcribe_batched(model or mock.Mock(), batched_model, mock.Mock(audio_path="a.mp3"), 4)

    def test_speech_only_chunks_are_batched_on_a_30s_grid_and_mapped_back(self) -> None:
        batched_model = mock.Mock()
        batched_model.transcribe.return_value = (
            [raw_segment(0.0, 1.0, "one"), raw_segment(30.0, 31.0, "two")],
            mock.Mock(),
        )

        segments, info, batch_size = self.run_batched(self.make_inference(), batched_model)

        batched_model.transcribe.assert_called_once()
        layout = batched_model.transcribe.call_args.args[0]
        kwargs = batched_model.transcribe.call_args.kwargs
        self.assertEqual(len(layout), 2 * 30 * 16_000)
        self.assertEqual(kwargs["clip_timestamps"], [{"start": 0.0, "end": 1.0}, {"start": 30.0, "end": 31.0}])
        self.assertEqual(kwargs["batch_size"], 4)
        self.assertFalse(kwargs["vad_filter"])
        self.assertFalse(kwargs["without_timestamps"])
        self.assertNotIn("vad_parameters", kwargs)
        # Chunk 2 starts at 2.88s and its speech starts 0.12s in, so "two" is at 3-4s in the original audio.
        self.assertEqual(segments, [Segment(0, 1_000, "one"), Segment(3_000, 4_000, "two")])
        self.assertEqual(info.duration_after_vad, 2.0)
        self.assertEqual(batch_size, 4)

    def test_pauses_inside_a_chunk_are_kept_short_and_mapped_back(self) -> None:
        inference = self.make_inference()
        default_side_effect = inference.vad_manager.get_speech_timestamps.side_effect

        def get_speech_timestamps(model_id, audio, sampling_rate, speech_probs, **kwargs):
            if len(audio) != 80_000 and speech_probs[0] > 0.5:
                # Chunk [0, 1.6s): speech 0-0.2s, 0.9-1.0s and 1.1-1.2s (pauses of 0.7s and 0.1s).
                return [{"start": 0, "end": 3_200}, {"start": 14_400, "end": 16_000}, {"start": 17_600, "end": 19_200}]
            return default_side_effect(model_id, audio, sampling_rate, speech_probs, **kwargs)

        inference.vad_manager.get_speech_timestamps.side_effect = get_speech_timestamps
        batched_model = mock.Mock()
        # In the concatenated chunk: speech 0-0.2s, kept silence 0.2-0.7s, speech 0.7-0.8s, 0.9-1.0s.
        batched_model.transcribe.return_value = (
            [raw_segment(0.0, 0.2, "a"), raw_segment(0.3, 1.0, "b"), raw_segment(30.0, 31.0, "two")],
            mock.Mock(),
        )

        segments, info, _batch_size = self.run_batched(inference, batched_model)

        layout = batched_model.transcribe.call_args.args[0]
        clips = batched_model.transcribe.call_args.kwargs["clip_timestamps"]
        self.assertEqual(clips[0], {"start": 0.0, "end": 1.0})
        self.assertTrue(np.all(layout[3_200:11_200] == 0))
        # "b" starts inside the kept silence, so it starts with the speech after it.
        self.assertEqual(segments[:2], [Segment(0, 200, "a"), Segment(900, 1_200, "b")])
        self.assertEqual(info.speech_segments, [{"start": 0, "end": 16_000}, {"start": 48_000, "end": 64_000}])
        batched_model.transcribe.assert_called_once()  # 0.4s of speech covered: no sequential retry

    def test_unreliable_chunks_are_decoded_again_sequentially(self) -> None:
        batched_model = mock.Mock()
        # Chunk 1 decodes fine; chunk 2 comes back empty although it has 1s of speech.
        batched_model.transcribe.return_value = ([raw_segment(0.0, 1.0, "one")], mock.Mock())
        model = mock.Mock()
        model.transcribe.return_value = ([raw_segment(0.12, 1.12, "two (sequential)")], mock.Mock(duration_after_vad=1))

        with self.assertLogs(infer_module.logger, "INFO") as logs:
            segments, _info, _batch_size = self.run_batched(self.make_inference(), batched_model, model)

        retry_logs = [line for line in logs.output if "decoded again sequentially" in line]
        self.assertEqual(len(retry_logs), 1)
        self.assertIn("no output for 1.0s of speech", retry_logs[0])
        model.transcribe.assert_called_once()
        chunk_audio = model.transcribe.call_args.args[0]
        self.assertEqual(len(chunk_audio), 25_600)  # the whole [2.88s, 4.48s) chunk
        self.assertTrue(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertEqual(segments, [Segment(0, 1_000, "one"), Segment(3_000, 4_000, "two (sequential)")])

    def test_oom_while_decoding_lazily_reduces_batch_size(self) -> None:
        def transcribe(audio, batch_size, **config):
            def segments():
                if batch_size > 2:
                    raise oom_error()
                yield raw_segment(0.0, 1.0, "ok")

            return segments(), mock.Mock()

        batched_model = mock.Mock()
        batched_model.transcribe.side_effect = transcribe

        segments, _info, batch_size = Inference.__new__(Inference)._transcribe_with_auto_batch_size(
            batched_model, np.zeros(16_000), {}, starting_batch_size=4
        )

        self.assertEqual([call.kwargs["batch_size"] for call in batched_model.transcribe.call_args_list], [4, 3, 2])
        self.assertEqual(batch_size, 2)
        self.assertEqual([segment.text for segment in segments], ["ok"])

    def test_probe_decodes_one_full_batch_per_attempt(self) -> None:
        def transcribe(audio, batch_size, clip_timestamps, **config):
            def segments():
                if batch_size > 2:
                    raise oom_error()
                yield raw_segment(0.0, 1.0, "ok")

            return segments(), mock.Mock()

        batched_model = mock.Mock()
        batched_model.transcribe.side_effect = transcribe
        inference = self.make_inference()

        with mock.patch.object(infer_module, "decode_audio", return_value=np.ones(5 * 16_000, dtype=np.float32)):
            batch_size = inference._find_executable_batch_size(batched_model, "a.mp3", 1, 4)

        self.assertEqual(batch_size, 2)
        for call in batched_model.transcribe.call_args_list:
            n = call.kwargs["batch_size"]
            self.assertEqual(len(call.args[0]), n * 30 * 16_000)
            self.assertEqual(len(call.kwargs["clip_timestamps"]), n)
            self.assertFalse(call.kwargs["vad_filter"])
        self.assertEqual([call.kwargs["batch_size"] for call in batched_model.transcribe.call_args_list], [4, 2])


if __name__ == "__main__":
    unittest.main()
