#!/usr/bin/env python3
"""
Inference script with custom VAD injection support
"""

import argparse
import code
import contextlib
import json
import logging
import os
import platform
import subprocess
import sys
import traceback
import unicodedata
from bisect import bisect_right
from collections import ChainMap, Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyjson5

# Import GPU/runtime-heavy deps defensively so `infer --help` still works on
# machines that don't have the required GPU runtime DLLs installed.
_FASTER_WHISPER_IMPORT_ERROR = None
_CTRANSLATE2_IMPORT_ERROR = None

try:
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    from faster_whisper.audio import decode_audio
except Exception as e:
    _FASTER_WHISPER_IMPORT_ERROR = e
    WhisperModel = None
    BatchedInferencePipeline = None
    decode_audio = None

try:
    import ctranslate2
except Exception as e:
    _CTRANSLATE2_IMPORT_ERROR = e
    ctranslate2 = None

# Import our VAD injection system
# Import modern i18n module for translations
from . import i18n_modern as i18n
from . import inject_vad, precomputed_speech_probs, uninject_vad
from .smart_split import (
    CHUNK_FINAL,
    CHUNK_FORCED,
    MAX_KEPT_GAP_S,
    AudioChunk,
    SpeechMap,
    choose_anchor,
    concat_speech,
    find_chunk_end,
    needs_sequential_retry,
    smooth_probs,
    split_at_pauses,
)
from .vad_manager import VadConfig, VadModelManager, resolve_neg_threshold

# Convenience imports
_ = i18n._
format_duration = i18n.format_duration
format_percentage = i18n.format_percentage

WHISPER_TASKS = ("transcribe", "translate")
WHISPER_SAMPLING_RATE = 16_000
MAX_SMART_CHUNK_DURATION_S = 30.0


def _normalize_whisper_task(task: Any) -> str:
    if not isinstance(task, str):
        raise ValueError(f"Whisper task must be one of {', '.join(WHISPER_TASKS)}")

    normalized = task.strip().lower()
    if normalized not in WHISPER_TASKS:
        raise ValueError(f"Invalid Whisper task '{task}'. Expected one of: {', '.join(WHISPER_TASKS)}")
    return normalized


def _require_ctranslate2():
    if ctranslate2 is None:
        raise RuntimeError(
            f"Failed to import ctranslate2. This build may be missing required GPU runtime libraries. "
            f"Original error: {_CTRANSLATE2_IMPORT_ERROR}"
        )
    return ctranslate2


def _require_faster_whisper():
    if WhisperModel is None or BatchedInferencePipeline is None or decode_audio is None:
        raise RuntimeError(
            f"Failed to import faster_whisper. This build may be missing required runtime libraries. "
            f"Original error: {_FASTER_WHISPER_IMPORT_ERROR}"
        )
    return WhisperModel, BatchedInferencePipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description=_("app.description"))
    parser.add_argument("--model_name_or_path", type=str, default="models", help=_("args.model_path"))
    parser.add_argument("--device", type=str, default="auto", help=_("args.device"))
    parser.add_argument("--compute_type", type=str, default="auto", help=_("args.compute_type"))
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=0,
        help="CPU threads for Whisper and VAD when running on CPU (0 = auto, about the physical core count)",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help=_("args.overwrite"))
    parser.add_argument(
        "--audio_suffixes",
        type=str,
        default="wav,flac,mp3",
        help=_("args.audio_extensions"),
    )
    parser.add_argument("--sub_formats", type=str, default="lrc,vtt", help=_("args.subtitle_formats"))
    parser.add_argument("--output_dir", type=str, default=None, help=_("args.output_dir"))
    parser.add_argument(
        "--task",
        type=str,
        choices=WHISPER_TASKS,
        default=None,
        help=_("args.task"),
    )
    parser.add_argument(
        "--generation_config",
        type=str,
        default="generation_config.json5",
        help=_("args.config_file"),
    )
    parser.add_argument("--log_level", type=str, default="DEBUG", help=_("args.log_level"))

    # Subtitle post-processing options
    parser.add_argument(
        "--merge_segments",
        dest="merge_segments",
        action="store_true",
        default=None,
        help="Enable segment merge post-processing (override config file)",
    )
    parser.add_argument(
        "--no_merge_segments",
        dest="merge_segments",
        action="store_false",
        default=None,
        help="Disable segment merge post-processing (override config file)",
    )
    parser.add_argument(
        "--merge_max_gap_ms",
        type=int,
        default=None,
        help="Max allowed gap (ms) between segments for merging (override config file)",
    )
    parser.add_argument(
        "--merge_max_duration_ms",
        type=int,
        default=None,
        help="Max duration (ms) of a merged segment (override config file)",
    )
    parser.add_argument(
        "--smart_split_with_vad",
        type=str,
        default=None,
        help="Enable smart outer VAD chunking before faster-whisper internal VAD (true/false)",
    )
    parser.add_argument(
        "--target_chunk_duration_s",
        type=float,
        default=None,
        help="Target duration for smart VAD chunks in seconds (override config file)",
    )

    # VAD parameter overrides (whisper_vad is always used)
    parser.add_argument("--vad_threshold", type=float, default=None, help=_("args.vad_threshold"))
    parser.add_argument(
        "--vad_min_speech_duration_ms",
        type=int,
        default=None,
        help=_("args.min_speech_duration"),
    )
    parser.add_argument(
        "--vad_min_silence_duration_ms",
        type=int,
        default=None,
        help=_("args.min_silence_duration"),
    )
    parser.add_argument("--vad_speech_pad_ms", type=int, default=None, help=_("args.speech_padding"))

    # Debug option for interactive console
    parser.add_argument(
        "--console",
        action="store_true",
        help="Launch interactive Python console for debugging",
    )

    # Batch inference options
    parser.add_argument(
        "--enable_batching",
        action="store_true",
        help="Enable batched inference for faster processing (requires more VRAM)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for batched inference (auto-detect if not specified)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum batch size to try when auto-detecting (default: 8)",
    )

    parser.add_argument("base_dirs", nargs=argparse.REMAINDER, help=_("args.directories"))
    return parser.parse_args()


def resolve_device(device: str) -> str:
    """Normalize the device name and resolve 'auto' to the device CTranslate2 will actually use."""
    # Normalize and accept friendly aliases.
    device = (device or "auto").strip().lower()
    if device in {"amd", "rocm", "hip"}:
        return "cuda"  # HIP builds still use the public device name "cuda".
    if device != "auto":
        return device

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible in {"", "-1"}:
        return "cpu"
    try:
        return "cuda" if _require_ctranslate2().get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def default_cpu_threads() -> int:
    """Approximate the physical core count (logical cores / 2 for SMT CPUs)."""
    return max(1, (os.cpu_count() or 8) // 2)


def select_best_compute_type(device: str) -> str:
    """
    Automatically select the best compute type based on device and available types.

    Preference order:
    - bfloat16 > float16 > int8 types > float32
    - Prefer int8 over float32 for better memory usage

    Args:
        device: The device to use ('cpu', 'cuda', or 'auto')

    Returns:
        The best available compute type for the device
    """
    ct2 = _require_ctranslate2()

    actual_device = resolve_device(device)
    if (device or "auto").strip().lower() == "auto":
        logger.info(_("info.auto_detected_device").format(device=actual_device))

    # Get supported compute types for the device.
    try:
        supported_types = ct2.get_supported_compute_types(actual_device)
    except Exception as e:
        logger.warning(_("warnings.compute_types_unavailable").format(device=actual_device, error=e))
        # Fallback to safe default
        return "int8" if actual_device == "cpu" else "float16"

    # Define preference order
    # Prefer bfloat16 > float16 > int8 types > float32
    preference_order = [
        "bfloat16",
        "float16",
        "int16",  # For CPU
        "int8_bfloat16",
        "int8_float16",
        "int8_float32",
        "int8",
        "float32",  # Least preferred due to memory usage
    ]

    # Select the best available type based on preference
    for compute_type in preference_order:
        if compute_type in supported_types:
            logger.info(_("info.auto_selected_compute_type").format(compute_type=compute_type, device=actual_device))
            return compute_type

    # If nothing matched (shouldn't happen), use a safe default
    default = "int8" if actual_device == "cpu" else "float16"
    logger.warning(_("warnings.no_preferred_compute_type").format(default=default))
    return default


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected boolean value, got {value!r}")


@dataclass
class Segment:
    start: int  # ms
    end: int  # ms
    text: str


@dataclass
class SmartSplitInput:
    audio: Any
    speech_probs: np.ndarray | None  # whole-file VAD frame probabilities, None if VAD is unavailable
    vad_segments: list[dict[str, Any]]  # outer VAD speech segments in samples
    duration_after_vad: float | None


@dataclass(frozen=True)
class SegmentMergeOptions:
    enabled: bool = True
    max_gap_ms: int = 2_000
    max_duration_ms: int = 20_000


@dataclass(frozen=True)
class SmartSplitOptions:
    enabled: bool = True
    target_chunk_duration_s: float = MAX_SMART_CHUNK_DURATION_S
    split_window_factor: float = 0.4


@dataclass(frozen=True)
class PauseSplitOptions:
    enabled: bool = True
    min_pause_ms: int = 1_500


DEFAULT_HALLUCINATION_BLOCKLIST = ("感谢您的视听", "感谢观看", "谢谢观看", "感谢收看", "谢谢收看", "完")


@dataclass(frozen=True)
class HallucinationFilterOptions:
    enabled: bool = True
    max_chars_per_second: float = 20.0
    min_chars: int = 10
    blocklist: tuple[str, ...] = DEFAULT_HALLUCINATION_BLOCKLIST


def _normalize_merge_text(text: str) -> str:
    return " ".join(text.strip().split())


def vad_segments_to_clip_timestamps(
    vad_segments: list[dict[str, Any]], sampling_rate: int = WHISPER_SAMPLING_RATE
) -> list[float]:
    timestamps: list[float] = []
    for segment in vad_segments:
        start = float(segment["start"]) / sampling_rate
        end = float(segment["end"]) / sampling_rate
        if end > start:
            timestamps.extend([start, end])
    return timestamps


def has_speech_between(vad_segments: list[dict[str, Any]], start_sample: int, end_sample: int) -> bool:
    return any(segment["start"] < end_sample and segment["end"] > start_sample for segment in vad_segments)


def _max_segment_end(start: int, end: int, max_duration_ms: int | None) -> int:
    if max_duration_ms is None or max_duration_ms <= 0:
        return end
    return min(end, start + max_duration_ms)


def enforce_segment_timeline(segments: list[Segment], max_duration_ms: int | None = None) -> list[Segment]:
    normalized: list[Segment] = []

    for segment in sorted((s for s in segments if s.text.strip()), key=lambda s: (s.start, s.end)):
        start = max(segment.start, normalized[-1].end if normalized else segment.start)
        end = _max_segment_end(start, segment.end, max_duration_ms)
        if end <= start:
            continue
        normalized.append(Segment(start=start, end=end, text=segment.text))

    return normalized


def merge_segments(segments: list[Segment], options: SegmentMergeOptions | None = None) -> list[Segment]:
    if options is None:
        options = SegmentMergeOptions()

    segments = [s for s in segments if s.text.strip()]
    segments.sort(key=lambda s: (s.start, s.end))
    if not options.enabled:
        return segments

    merged: list[Segment] = []

    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]

        gap_ms = seg.start - last.end
        if gap_ms > options.max_gap_ms:
            merged.append(seg)
            continue

        merged_duration_ms = seg.end - last.start
        if merged_duration_ms > options.max_duration_ms:
            merged.append(seg)
            continue

        last_norm = _normalize_merge_text(last.text)
        seg_norm = _normalize_merge_text(seg.text)

        if seg_norm.startswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        if last_norm.startswith(seg_norm) or last_norm.endswith(seg_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=last.text)
            continue

        if seg_norm.endswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        merged.append(seg)

    return merged


def split_segments_at_pauses(
    segments: list[Segment],
    vad_segments: list[dict[str, Any]],
    options: PauseSplitOptions,
    sampling_rate: int = WHISPER_SAMPLING_RATE,
) -> list[Segment]:
    """Split (or trim) segments that run across pauses of at least options.min_pause_ms in the VAD speech."""
    if not options.enabled or not vad_segments:
        return segments

    pieces = sorted((s["start"] / sampling_rate, s["end"] / sampling_rate) for s in vad_segments)
    piece_ends = [end for _start, end in pieces]
    result: list[Segment] = []
    for segment in segments:
        start_s, end_s = segment.start / 1_000, segment.end / 1_000
        first = bisect_right(piece_ends, start_s)
        overlapping = []
        for piece in pieces[first:]:
            if piece[0] >= end_s:
                break
            overlapping.append(piece)
        parts = split_at_pauses(start_s, end_s, segment.text, overlapping, min_pause_s=options.min_pause_ms / 1_000)
        if len(parts) > 1:
            logger.debug(
                "Split at pauses: [%s --> %s] %s => %s",
                SubWriter.lrc_timestamp(segment.start),
                SubWriter.lrc_timestamp(segment.end),
                segment.text,
                " | ".join(text for _start, _end, text in parts),
            )
        for part_start, part_end, text in parts:
            result.append(Segment(start=int(round(part_start * 1_000)), end=int(round(part_end * 1_000)), text=text))
    return result


def _normalize_filter_text(text: str) -> str:
    """Text without whitespace and punctuation, for counting characters and matching the blocklist."""
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P")).casefold()


def filter_hallucinations(segments: list[Segment], options: HallucinationFilterOptions) -> list[Segment]:
    """Drop segments that are known Whisper hallucinations or have more text than could be spoken in their time."""
    if not options.enabled:
        return segments

    blocklist = {_normalize_filter_text(text) for text in options.blocklist}
    kept: list[Segment] = []
    for segment in segments:
        normalized = _normalize_filter_text(segment.text)
        duration_s = max(segment.end - segment.start, 1) / 1_000
        if normalized in blocklist:
            reason = "blocklist"
        elif len(normalized) >= options.min_chars and len(normalized) / duration_s > options.max_chars_per_second:
            reason = f"{len(normalized) / duration_s:.1f} chars/s"
        else:
            kept.append(segment)
            continue
        logger.info(
            "Dropped likely hallucination (%s): [%s --> %s] %s",
            reason,
            SubWriter.lrc_timestamp(segment.start),
            SubWriter.lrc_timestamp(segment.end),
            segment.text,
        )
    return kept


def _log_decoded_segment(segment: Segment, raw: Any) -> None:
    """Debug-log a decoded segment with Whisper's confidence values, for tuning the hallucination filter."""
    logger.debug(
        "[%s --> %s] %s (avg_logprob=%s, no_speech_prob=%s, compression_ratio=%s, temperature=%s)",
        SubWriter.lrc_timestamp(segment.start),
        SubWriter.lrc_timestamp(segment.end),
        segment.text,
        getattr(raw, "avg_logprob", None),
        getattr(raw, "no_speech_prob", None),
        getattr(raw, "compression_ratio", None),
        getattr(raw, "temperature", None),
    )


class SubWriter:
    @classmethod
    def txt(cls, segments: list[Segment], path: str):
        lines = []
        for _idx, segment in enumerate(segments):
            lines.append(f"{segment.text}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def lrc(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            start_ts = cls.lrc_timestamp(segment.start)
            end_es = cls.lrc_timestamp(segment.end)
            lines.append(f"[{start_ts}]{segment.text}\n")
            if idx != len(segments) - 1:
                next_start = segments[idx + 1].start
                if next_start is not None and end_es == cls.lrc_timestamp(next_start):
                    continue
            lines.append(f"[{end_es}]\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def lrc_timestamp(ms: int) -> str:
        m = ms // 60_000
        ms = ms - m * 60_000
        s = ms // 1_000
        ms = ms - s * 1_000
        ms = ms // 10
        return f"{m:02d}:{s:02d}.{ms:02d}"

    @classmethod
    def vtt(cls, segments: list[Segment], path: str):
        lines = ["WebVTT\n\n"]
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.vtt_timestamp(segment.start)} --> {cls.vtt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def vtt_timestamp(cls, ms: int):
        return cls._timestamp(ms, ".")

    @classmethod
    def srt(cls, segments: list[Segment], path: str):
        lines = []
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.srt_timestamp(segment.start)} --> {cls.srt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def srt_timestamp(cls, ms: int):
        return cls._timestamp(ms, ",")

    @classmethod
    def _timestamp(cls, ms: int, delim: str):
        h = ms // 3600_000
        ms -= h * 3600_000
        m = ms // 60_000
        ms -= m * 60_000
        s = ms // 1_000
        ms -= s * 1_000
        return f"{h:02d}:{m:02d}:{s:02d}{delim}{ms:03d}"


@dataclass
class InferenceTask:
    audio_path: str
    sub_prefix: str
    sub_formats: list[str]


logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(log_handler)


class Inference:
    sub_writers = {
        "lrc": SubWriter.lrc,
        "srt": SubWriter.srt,
        "vtt": SubWriter.vtt,
        "txt": SubWriter.txt,
    }

    def __init__(self, args):
        self.args = args
        self.model_name_or_path = args.model_name_or_path
        self.vad_injected = False
        self.vad_manager = None
        self.device = (args.device or "auto").strip().lower()
        if self.device in {"amd", "rocm", "hip"}:
            # CTranslate2 HIP backend still uses the public device name "cuda".
            self.device = "cuda"
        # Auto-select compute type if 'auto' or 'default' is specified
        if args.compute_type in ["auto", "default"]:
            self.compute_type = select_best_compute_type(self.device)
        else:
            self.compute_type = args.compute_type

        # CTranslate2 only uses 4 CPU threads by default; use the physical cores on CPU.
        requested_cpu_threads = getattr(args, "cpu_threads", 0) or 0
        if requested_cpu_threads > 0:
            self.cpu_threads = requested_cpu_threads
        elif resolve_device(self.device) == "cpu":
            self.cpu_threads = default_cpu_threads()
        else:
            self.cpu_threads = 0
        if self.cpu_threads:
            logger.info("CPU threads: %s", self.cpu_threads)

        # Batch inference settings
        self.enable_batching = args.enable_batching
        self.batch_size = args.batch_size if args.batch_size else 0
        self.max_batch_size = args.max_batch_size

        self.overwrite = args.overwrite
        self.output_dir = args.output_dir
        if self.output_dir:
            if not os.path.isabs(self.output_dir):
                self.output_dir = os.path.join(os.getcwd(), self.output_dir)
            logger.info(_("info.output_dir", output_dir=self.output_dir))
        self.audio_suffixes = {k: True for k in args.audio_suffixes.split(",")}
        self.sub_formats = []
        for k in args.sub_formats.split(","):
            if k not in self.sub_writers:
                raise ValueError(_("warnings.unknown_format", format=k))
            self.sub_formats.append(k)

        # Load generation config
        (
            self.generation_config,
            self.segment_merge_options,
            self.smart_split_options,
            self.pause_split_options,
            self.hallucination_filter_options,
        ) = self._load_generation_config(args)

        # Setup VAD injection if requested
        self._setup_vad_injection(args)

        logger.info(_("info.generation_config", config=self.generation_config))
        logger.info(
            "Segment merge: enabled=%s, max_gap_ms=%s, max_duration_ms=%s",
            self.segment_merge_options.enabled,
            self.segment_merge_options.max_gap_ms,
            self.segment_merge_options.max_duration_ms,
        )
        logger.info(
            "Smart VAD split: enabled=%s, target_chunk_duration_s=%s",
            self.smart_split_options.enabled,
            self.smart_split_options.target_chunk_duration_s,
        )
        logger.info(
            "Pause split: enabled=%s, min_pause_ms=%s",
            self.pause_split_options.enabled,
            self.pause_split_options.min_pause_ms,
        )
        logger.info(
            "Hallucination filter: enabled=%s, max_chars_per_second=%s, min_chars=%s, blocklist=%s",
            self.hallucination_filter_options.enabled,
            self.hallucination_filter_options.max_chars_per_second,
            self.hallucination_filter_options.min_chars,
            list(self.hallucination_filter_options.blocklist),
        )

    def _load_generation_config(
        self, args
    ) -> tuple[dict[str, Any], SegmentMergeOptions, SmartSplitOptions, PauseSplitOptions, HallucinationFilterOptions]:
        """Load and process generation configuration"""
        # Default config
        config: dict[str, Any] = {
            "language": "ja",
            "task": "translate",
            "vad_filter": True,
        }

        segment_merge_options = SegmentMergeOptions()
        smart_split_options = SmartSplitOptions()
        pause_split_options = PauseSplitOptions()
        hallucination_filter_options = HallucinationFilterOptions()

        # Load from file if exists
        if os.path.exists(args.generation_config):
            with open(args.generation_config, encoding="utf-8") as f:
                file_config = pyjson5.decode_io(f)
                file_segment_merge = file_config.pop("segment_merge", None)
                if isinstance(file_segment_merge, dict):
                    segment_merge_options = SegmentMergeOptions(
                        enabled=bool(file_segment_merge.get("enabled", segment_merge_options.enabled)),
                        max_gap_ms=int(file_segment_merge.get("max_gap_ms", segment_merge_options.max_gap_ms)),
                        max_duration_ms=int(
                            file_segment_merge.get("max_duration_ms", segment_merge_options.max_duration_ms)
                        ),
                    )
                file_smart_split = file_config.pop("smart_split_with_vad", None)
                file_target_chunk_duration = file_config.pop("target_chunk_duration_s", None)
                smart_split_options = SmartSplitOptions(
                    enabled=_coerce_bool(file_smart_split, default=smart_split_options.enabled),
                    target_chunk_duration_s=(
                        float(file_target_chunk_duration)
                        if file_target_chunk_duration is not None
                        else smart_split_options.target_chunk_duration_s
                    ),
                    split_window_factor=smart_split_options.split_window_factor,
                )
                file_pause_split = file_config.pop("pause_split", None)
                if isinstance(file_pause_split, dict):
                    pause_split_options = PauseSplitOptions(
                        enabled=_coerce_bool(file_pause_split.get("enabled"), default=pause_split_options.enabled),
                        min_pause_ms=int(file_pause_split.get("min_pause_ms", pause_split_options.min_pause_ms)),
                    )
                file_hallucination_filter = file_config.pop("hallucination_filter", None)
                if isinstance(file_hallucination_filter, dict):
                    defaults = hallucination_filter_options
                    hallucination_filter_options = HallucinationFilterOptions(
                        enabled=_coerce_bool(file_hallucination_filter.get("enabled"), default=defaults.enabled),
                        max_chars_per_second=float(
                            file_hallucination_filter.get("max_chars_per_second", defaults.max_chars_per_second)
                        ),
                        min_chars=int(file_hallucination_filter.get("min_chars", defaults.min_chars)),
                        blocklist=tuple(
                            str(text) for text in file_hallucination_filter.get("blocklist", defaults.blocklist)
                        ),
                    )
                config = dict(**ChainMap(file_config, config))

        config["task"] = _normalize_whisper_task(args.task if args.task is not None else config.get("task"))

        # Process VAD parameters from config file
        if "vad_parameters" in config:
            vad_params = config.pop("vad_parameters")

            # Convert to VadOptions format
            vad_options = {}

            # Map common parameters
            if "threshold" in vad_params:
                vad_options["threshold"] = vad_params["threshold"]
            if "neg_threshold" in vad_params:
                vad_options["neg_threshold"] = vad_params["neg_threshold"]
            if "min_speech_duration_ms" in vad_params:
                vad_options["min_speech_duration_ms"] = vad_params["min_speech_duration_ms"]
            if "max_speech_duration_s" in vad_params:
                vad_options["max_speech_duration_s"] = vad_params["max_speech_duration_s"]
            if "min_silence_duration_ms" in vad_params:
                vad_options["min_silence_duration_ms"] = vad_params["min_silence_duration_ms"]
            if "speech_pad_ms" in vad_params:
                vad_options["speech_pad_ms"] = vad_params["speech_pad_ms"]

            config["vad_parameters"] = vad_options

        # Override with command line arguments
        if args.vad_threshold is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["threshold"] = args.vad_threshold

        if args.vad_min_speech_duration_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["min_speech_duration_ms"] = args.vad_min_speech_duration_ms

        if args.vad_min_silence_duration_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["min_silence_duration_ms"] = args.vad_min_silence_duration_ms

        if args.vad_speech_pad_ms is not None:
            if "vad_parameters" not in config:
                config["vad_parameters"] = {}
            config["vad_parameters"]["speech_pad_ms"] = args.vad_speech_pad_ms

        # Override segment merge options with command line arguments
        segment_merge_options = SegmentMergeOptions(
            enabled=args.merge_segments if args.merge_segments is not None else segment_merge_options.enabled,
            max_gap_ms=args.merge_max_gap_ms if args.merge_max_gap_ms is not None else segment_merge_options.max_gap_ms,
            max_duration_ms=(
                args.merge_max_duration_ms
                if args.merge_max_duration_ms is not None
                else segment_merge_options.max_duration_ms
            ),
        )

        smart_split_options = SmartSplitOptions(
            enabled=(
                _coerce_bool(args.smart_split_with_vad, default=smart_split_options.enabled)
                if args.smart_split_with_vad is not None
                else smart_split_options.enabled
            ),
            target_chunk_duration_s=(
                args.target_chunk_duration_s
                if args.target_chunk_duration_s is not None
                else smart_split_options.target_chunk_duration_s
            ),
            split_window_factor=smart_split_options.split_window_factor,
        )
        if smart_split_options.target_chunk_duration_s <= 0:
            raise ValueError("target_chunk_duration_s must be greater than 0")
        if smart_split_options.target_chunk_duration_s > MAX_SMART_CHUNK_DURATION_S:
            smart_split_options = SmartSplitOptions(
                enabled=smart_split_options.enabled,
                target_chunk_duration_s=MAX_SMART_CHUNK_DURATION_S,
                split_window_factor=smart_split_options.split_window_factor,
            )

        return config, segment_merge_options, smart_split_options, pause_split_options, hallucination_filter_options

    def _vad_progress_callback(self, chunk_idx, total_chunks, device):
        """Progress callback for VAD processing."""
        progress_pct = (chunk_idx / total_chunks) * 100
        # Use carriage return to update the same line
        print(
            "\r  "
            + _(
                "progress.vad",
                current=chunk_idx,
                total=total_chunks,
                percent=progress_pct,
                device=device,
            ),
            end="",
            flush=True,
        )
        if chunk_idx == total_chunks:
            print()  # New line when done

    def _setup_vad_injection(self, args):
        """Setup whisper_vad injection - always enforced"""
        # Always use whisper_vad model
        vad_model = "whisper_vad"

        logger.info(_("info.initializing_vad"))

        # Create VAD config with progress callback
        vad_config = VadConfig(default_model=vad_model)

        # Apply VAD parameters from generation config
        if "vad_parameters" in self.generation_config:
            vad_params = self.generation_config["vad_parameters"]
            if "threshold" in vad_params:
                vad_config.threshold = vad_params["threshold"]
            if "neg_threshold" in vad_params:
                vad_config.neg_threshold = vad_params["neg_threshold"]
            if "min_speech_duration_ms" in vad_params:
                vad_config.min_speech_duration_ms = vad_params["min_speech_duration_ms"]
            if "max_speech_duration_s" in vad_params:
                vad_config.max_speech_duration_s = vad_params["max_speech_duration_s"]
            if "min_silence_duration_ms" in vad_params:
                vad_config.min_silence_duration_ms = vad_params["min_silence_duration_ms"]
            if "speech_pad_ms" in vad_params:
                vad_config.speech_pad_ms = vad_params["speech_pad_ms"]

        # Load ONNX VAD configuration from metadata
        vad_metadata_path = "models/whisper_vad_metadata.json"
        vad_config.onnx_model_path = "models/whisper_vad.onnx"
        vad_config.onnx_metadata_path = vad_metadata_path

        # Read model configuration from metadata JSON if it exists
        if os.path.exists(vad_metadata_path):
            try:
                with open(vad_metadata_path) as f:
                    metadata = json.load(f)

                # Load model configuration from metadata
                vad_config.whisper_model_name = metadata.get("whisper_model_name", "openai/whisper-base")
                vad_config.frame_duration_ms = metadata.get("frame_duration_ms", 20)
                vad_config.chunk_duration_ms = metadata.get("total_duration_ms", 30000)

                logger.info(_("warnings.loaded_vad_config", path=vad_metadata_path))
            except Exception as e:
                logger.warning(_("warnings.failed_load_vad", path=vad_metadata_path, error=e))
                logger.warning(_("warnings.using_default_vad"))
                # Fallback to defaults
                vad_config.whisper_model_name = "openai/whisper-base"
                vad_config.frame_duration_ms = 20
                vad_config.chunk_duration_ms = 30000
        else:
            # Use defaults if metadata file doesn't exist
            logger.warning(_("warnings.vad_file_not_found", path=vad_metadata_path))
            logger.warning(_("warnings.using_default_vad"))
            vad_config.whisper_model_name = "openai/whisper-base"
            vad_config.frame_duration_ms = 20
            vad_config.chunk_duration_ms = 30000

        # Hardcoded runtime configuration
        vad_config.force_cpu = False
        vad_config.num_threads = self.cpu_threads or default_cpu_threads()

        # Inject VAD with progress callback
        self.vad_manager = VadModelManager(
            config=vad_config,
            ttl=vad_config.ttl,
            progress_callback=self._vad_progress_callback,
        )
        inject_vad(
            model_id=vad_model,
            config=vad_config,
            progress_callback=self._vad_progress_callback,
            manager=self.vad_manager,
        )
        self.vad_injected = True
        logger.info(_("info.vad_activated", threshold=vad_config.threshold))

    def _prepare_transcription(self, audio_path: str) -> tuple[Any, dict[str, Any], float | None]:
        config = dict(self.generation_config)

        if self.smart_split_options.enabled or not config.get("vad_filter") or "clip_timestamps" in config:
            return audio_path, config, None

        if self.vad_manager is None:
            return audio_path, config, None

        audio = decode_audio(audio_path, sampling_rate=WHISPER_SAMPLING_RATE)
        vad_parameters = config.get("vad_parameters") or {}
        vad_segments = self.vad_manager.get_speech_timestamps(
            model_id="whisper_vad",
            audio=audio,
            sampling_rate=WHISPER_SAMPLING_RATE,
            **vad_parameters,
        )
        duration_after_vad = sum(segment["end"] - segment["start"] for segment in vad_segments) / WHISPER_SAMPLING_RATE

        config["vad_filter"] = False
        config["clip_timestamps"] = vad_segments_to_clip_timestamps(vad_segments, WHISPER_SAMPLING_RATE)
        config.setdefault("beam_size", 1)
        config.setdefault("condition_on_previous_text", False)

        return audio, config, duration_after_vad

    def _prepare_smart_split(self, audio_path: str) -> SmartSplitInput:
        audio = decode_audio(audio_path, sampling_rate=WHISPER_SAMPLING_RATE)
        if self.vad_manager is None:
            return SmartSplitInput(audio, None, [], None)

        # Run the VAD model once; chunk planning and the per-chunk VAD inside faster_whisper reuse these.
        speech_probs = self.vad_manager.get_speech_probs("whisper_vad", audio, WHISPER_SAMPLING_RATE)
        vad_parameters = self.generation_config.get("vad_parameters") or {}
        vad_segments = self.vad_manager.get_speech_timestamps(
            model_id="whisper_vad",
            audio=audio,
            sampling_rate=WHISPER_SAMPLING_RATE,
            speech_probs=speech_probs,
            **vad_parameters,
        )
        duration_after_vad = sum(segment["end"] - segment["start"] for segment in vad_segments) / WHISPER_SAMPLING_RATE
        return SmartSplitInput(audio, speech_probs, vad_segments, duration_after_vad)

    def _smart_split_frame_params(self) -> tuple[int, int, float]:
        """(frame_samples, max_frames, pause_threshold) for planning chunks on VAD frames."""
        assert self.vad_manager is not None
        frame_ms = self.vad_manager.frame_duration_ms
        frame_samples = WHISPER_SAMPLING_RATE * frame_ms // 1000
        target_s = min(self.smart_split_options.target_chunk_duration_s, MAX_SMART_CHUNK_DURATION_S)
        max_frames = max(1, int(target_s * 1000 // frame_ms))
        vad_parameters = self.generation_config.get("vad_parameters") or {}
        threshold = vad_parameters.get("threshold", VadConfig.threshold)
        pause_threshold = resolve_neg_threshold(threshold, vad_parameters.get("neg_threshold"))
        return frame_samples, max_frames, pause_threshold

    def _next_chunk(self, smoothed: np.ndarray, start_frame: int, index: int, audio_len: int) -> tuple[AudioChunk, int]:
        """Plan the chunk starting at start_frame; returns it with its (exclusive) end frame."""
        frame_samples, max_frames, pause_threshold = self._smart_split_frame_params()
        end_frame, kind = find_chunk_end(
            smoothed,
            start_frame,
            max_frames=max_frames,
            split_window_factor=self.smart_split_options.split_window_factor,
            pause_threshold=pause_threshold,
            min_search_frames=WHISPER_SAMPLING_RATE // frame_samples,  # 1s
        )
        start_sample = start_frame * frame_samples
        end_sample = audio_len if kind == CHUNK_FINAL else min(audio_len, end_frame * frame_samples)
        chunk = AudioChunk(index, start_sample / WHISPER_SAMPLING_RATE, end_sample / WHISPER_SAMPLING_RATE, kind)
        return chunk, end_frame

    def _transcribe_chunk(
        self, model, audio, chunk: AudioChunk, chunk_probs: np.ndarray | None, config: dict[str, Any]
    ) -> tuple[list[Segment], float | None]:
        """Transcribe one contiguous chunk; returns segments in absolute ms (clipped to the chunk) and its VAD duration."""
        start_sample = int(round(chunk.start * WHISPER_SAMPLING_RATE))
        end_sample = int(round(chunk.end * WHISPER_SAMPLING_RATE))
        logger.debug(
            "Smart VAD chunk %s (%s): %s --> %s",
            chunk.index + 1,
            chunk.kind,
            SubWriter.srt_timestamp(int(round(chunk.start * 1_000))),
            SubWriter.srt_timestamp(int(round(chunk.end * 1_000))),
        )
        with contextlib.ExitStack() as stack:
            if chunk_probs is not None:
                stack.enter_context(precomputed_speech_probs(chunk_probs))
            segments_iter, info = model.transcribe(audio[start_sample:end_sample], **config)
            # Decoding is lazy; consume it inside the context so any deferred VAD call still sees the probs.
            raw_segments = list(segments_iter)

        chunk_offset_ms = int(round(chunk.start * 1_000))
        chunk_end_ms = int(round(chunk.end * 1_000))
        segments: list[Segment] = []
        for _segment in raw_segments:
            start = chunk_offset_ms + int(round(_segment.start * 1_000))
            end = min(chunk_offset_ms + int(round(_segment.end * 1_000)), chunk_end_ms)
            if start >= chunk_end_ms or end <= start:
                continue
            segment = Segment(start=start, end=end, text=_segment.text.strip())
            segments.append(segment)
            _log_decoded_segment(segment, _segment)
        duration_after_vad = float(info.duration_after_vad) if hasattr(info, "duration_after_vad") else None
        return segments, duration_after_vad

    def _smart_chunk_config(self) -> dict[str, Any]:
        config = dict(self.generation_config)
        config.pop("clip_timestamps", None)
        config["vad_filter"] = bool(config.get("vad_filter", True))
        config.setdefault("beam_size", 1)
        config.setdefault("condition_on_previous_text", False)
        return config

    def _transcribe_smart_chunks(self, model, task: InferenceTask) -> tuple[list[Segment], Any]:
        split_input = self._prepare_smart_split(task.audio_path)
        audio = split_input.audio
        duration = len(audio) / WHISPER_SAMPLING_RATE
        config = self._smart_chunk_config()

        if split_input.duration_after_vad == 0:
            return [], SimpleNamespace(duration=duration, duration_after_vad=0)

        if split_input.speech_probs is None:
            # No VAD model: transcribe the file as one piece and let faster_whisper handle it.
            whole_segments, inner_duration = self._transcribe_chunk(
                model, audio, AudioChunk(0, 0.0, duration), None, config
            )
            return whole_segments, SimpleNamespace(duration=duration, duration_after_vad=inner_duration or duration)

        speech_probs = split_input.speech_probs
        frame_samples = self._smart_split_frame_params()[0]
        smoothed = smooth_probs(speech_probs)

        segments: list[Segment] = []
        stats: Counter[str] = Counter()
        inner_duration_after_vad = 0.0
        has_inner_duration = False
        start_frame = 0
        chunk_index = 0

        while start_frame < len(speech_probs):
            chunk, end_frame = self._next_chunk(smoothed, start_frame, chunk_index, len(audio))
            chunk_index += 1
            stats[chunk.kind] += 1

            if not has_speech_between(split_input.vad_segments, start_frame * frame_samples, end_frame * frame_samples):
                stats["skipped"] += 1
                start_frame = end_frame
                continue

            chunk_segments, chunk_duration = self._transcribe_chunk(
                model, audio, chunk, speech_probs[start_frame:end_frame], config
            )
            if chunk_duration is not None:
                inner_duration_after_vad += chunk_duration
                has_inner_duration = True

            next_start_frame = end_frame
            if chunk.kind == CHUNK_FORCED:
                # The cut may have split a word: keep only the segments that ended well before it and
                # decode the rest again as part of the next chunk.
                anchor = choose_anchor(
                    [segment.end / 1_000 for segment in chunk_segments],
                    chunk.start,
                    chunk.end,
                    smoothed,
                    frame_samples / WHISPER_SAMPLING_RATE,
                )
                if anchor is not None:
                    next_start_frame, keep_until_s = anchor
                    keep_until_ms = int(round(keep_until_s * 1_000))
                    chunk_segments = [segment for segment in chunk_segments if segment.end <= keep_until_ms]
                    stats["reanchored"] += 1
            segments.extend(chunk_segments)
            start_frame = next_start_frame

        logger.info(
            "Smart VAD split: chunks=%s pause=%s forced=%s reanchored=%s final=%s skipped=%s",
            chunk_index,
            stats["pause"],
            stats["forced"],
            stats["reanchored"],
            stats["final"],
            stats["skipped"],
        )
        duration_after_vad = inner_duration_after_vad if has_inner_duration else split_input.duration_after_vad
        return segments, SimpleNamespace(
            duration=duration, duration_after_vad=duration_after_vad, speech_segments=split_input.vad_segments
        )

    def _batched_config(self) -> dict[str, Any]:
        """Config for BatchedInferencePipeline.transcribe when we provide the (speech-only) clips ourselves."""
        config = dict(self.generation_config)
        for key in ("clip_timestamps", "vad_parameters"):
            config.pop(key, None)
        config["vad_filter"] = False
        config.setdefault("beam_size", 1)
        config.setdefault("condition_on_previous_text", False)
        # Don't rely on the patched default: subtitle timestamps are needed.
        config.setdefault("without_timestamps", False)
        return config

    def _transcribe_batched(
        self, model, batched_model, task: InferenceTask, batch_size: int
    ) -> tuple[list[Segment], Any, int]:
        """
        Speed mode: decode the smart split chunks in batches.

        Chunks are planned exactly like the sequential path (without re-anchoring, which needs the
        previous chunk's result). Each chunk's speech is concatenated with the same per-chunk VAD that
        faster_whisper would apply, keeping up to MAX_KEPT_GAP_S of each pause so Whisper still ends
        segments there, and the speech-only chunks are laid out on a 30s grid so the
        batched pipeline decodes one chunk per window. Batched decoding has no temperature fallback
        and no seek, so chunks whose result looks unreliable are decoded again sequentially.

        Returns (segments, info, batch size actually used).
        """
        split_input = self._prepare_smart_split(task.audio_path)
        audio = split_input.audio
        duration = len(audio) / WHISPER_SAMPLING_RATE
        if split_input.duration_after_vad == 0 or split_input.speech_probs is None:
            return [], SimpleNamespace(duration=duration, duration_after_vad=0), batch_size
        assert self.vad_manager is not None

        speech_probs = split_input.speech_probs
        frame_samples = self._smart_split_frame_params()[0]
        smoothed = smooth_probs(speech_probs)
        vad_parameters = self.generation_config.get("vad_parameters") or {}
        stats: Counter[str] = Counter()

        # Plan all chunks up front and cut each one down to its speech.
        prepared: list[tuple[AudioChunk, int, int, np.ndarray, SpeechMap]] = []
        start_frame = 0
        chunk_index = 0
        while start_frame < len(speech_probs):
            chunk, end_frame = self._next_chunk(smoothed, start_frame, chunk_index, len(audio))
            chunk_index += 1
            stats[chunk.kind] += 1
            start_sample = start_frame * frame_samples
            end_sample = int(round(chunk.end * WHISPER_SAMPLING_RATE))
            chunk_audio = audio[start_sample:end_sample]
            inner_segments = []
            if has_speech_between(split_input.vad_segments, start_sample, end_sample):
                inner_segments = self.vad_manager.get_speech_timestamps(
                    model_id="whisper_vad",
                    audio=chunk_audio,
                    sampling_rate=WHISPER_SAMPLING_RATE,
                    speech_probs=speech_probs[start_frame:end_frame],
                    **vad_parameters,
                )
            speech, speech_map = concat_speech(
                chunk_audio, inner_segments, WHISPER_SAMPLING_RATE, max_gap_s=MAX_KEPT_GAP_S
            )
            if len(speech) == 0:
                stats["skipped"] += 1
            else:
                prepared.append((chunk, start_frame, end_frame, speech, speech_map))
            start_frame = end_frame

        config = self._batched_config()
        sequential_config = self._smart_chunk_config()
        compression_ratio_threshold = self.generation_config.get("compression_ratio_threshold", 2.4)
        log_prob_threshold = self.generation_config.get("log_prob_threshold", -1.0)
        window_s = MAX_SMART_CHUNK_DURATION_S
        window_samples = int(window_s * WHISPER_SAMPLING_RATE)
        segments: list[Segment] = []

        group_start = 0
        while group_start < len(prepared):
            group = prepared[group_start : group_start + batch_size]
            layout = np.zeros(len(group) * window_samples, dtype=np.float32)
            clips = []
            for k, (_chunk, _f0, _f1, speech, _map) in enumerate(group):
                layout[k * window_samples : k * window_samples + len(speech)] = speech
                clips.append({"start": k * window_s, "end": k * window_s + len(speech) / WHISPER_SAMPLING_RATE})
            raw_segments, _info, batch_size = self._transcribe_with_auto_batch_size(
                batched_model, layout, {**config, "clip_timestamps": clips}, starting_batch_size=batch_size
            )

            by_chunk: list[list[Any]] = [[] for _ in group]
            for raw in raw_segments:
                by_chunk[min(len(group) - 1, max(0, int(raw.start // window_s)))].append(raw)

            for k, (chunk, f0, f1, _speech, speech_map) in enumerate(group):
                spans = [(raw.start - k * window_s, raw.end - k * window_s) for raw in by_chunk[k]]
                if needs_sequential_retry(
                    [(speech_map.to_speech_time(start), speech_map.to_speech_time(end)) for start, end in spans],
                    [raw.compression_ratio for raw in by_chunk[k]],
                    [raw.avg_logprob for raw in by_chunk[k]],
                    speech_map.duration_s,
                    compression_ratio_threshold=compression_ratio_threshold,
                    log_prob_threshold=log_prob_threshold,
                ):
                    stats["batched_fallback"] += 1
                    chunk_segments, _duration = self._transcribe_chunk(
                        model, audio, chunk, speech_probs[f0:f1], sequential_config
                    )
                    segments.extend(chunk_segments)
                    continue

                for raw, (local_start, local_end) in zip(by_chunk[k], spans, strict=True):
                    start_ms = int(round((chunk.start + speech_map.to_original(local_start)) * 1_000))
                    end_ms = int(round((chunk.start + speech_map.to_original(local_end, is_end=True)) * 1_000))
                    end_ms = min(end_ms, int(round(chunk.end * 1_000)))
                    if end_ms > start_ms:
                        segment = Segment(start=start_ms, end=end_ms, text=raw.text.strip())
                        segments.append(segment)
                        _log_decoded_segment(segment, raw)
            group_start += len(group)

        logger.info(
            "Smart VAD split (batched): chunks=%s pause=%s forced=%s final=%s skipped=%s batched_fallback=%s",
            chunk_index,
            stats["pause"],
            stats["forced"],
            stats["final"],
            stats["skipped"],
            stats["batched_fallback"],
        )
        duration_after_vad = sum(speech_map.duration_s for *_rest, speech_map in prepared)
        info = SimpleNamespace(
            duration=duration, duration_after_vad=duration_after_vad, speech_segments=split_input.vad_segments
        )
        return segments, info, batch_size

    def _transcribe_standard(self, model, audio_input, transcription_config):
        return model.transcribe(audio_input, **transcription_config)

    def _log_duration(self, duration: float, duration_after_vad: float) -> None:
        if duration == duration_after_vad or duration_after_vad == 0:
            logger.info(_("info.duration", duration=format_duration(duration)))
            return

        rate = duration_after_vad / duration
        logger.info(
            _(
                "info.duration_filtered",
                original=format_duration(duration),
                filtered=format_duration(duration_after_vad),
                percent=format_percentage(rate),
            )
        )

    def _should_use_smart_split(self) -> bool:
        return bool(self.smart_split_options.enabled and self.generation_config.get("vad_filter", True))

    def generates(self, base_dirs):
        if len(base_dirs) == 0:
            logger.warning(_("warnings.provide_directories"))
            return

        tasks = self._scan(base_dirs)
        if len(tasks) == 0:
            logger.info(_("info.no_files_found"))
            return

        logger.info(_("tasks.processing", count=len(tasks), task=self.generation_config["task"]))
        logger.info(_("info.loading_whisper"))

        try:
            WhisperModelCls, BatchedInferencePipelineCls = _require_faster_whisper()

            model = WhisperModelCls(
                self.model_name_or_path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
            logger.info(_("info.model_precision").format(precision=self.compute_type, device=self.device))

            # Setup batched inference if enabled
            batched_model = None
            batch_size_to_use = self.batch_size

            if self.enable_batching:
                try:
                    batched_model = BatchedInferencePipelineCls(model=model)

                    # Auto-detect batch size if not specified
                    if batch_size_to_use == 0 and len(tasks) > 0:
                        # Use the first audio file as sample for testing
                        batch_size_to_use = self._find_executable_batch_size(
                            batched_model,
                            tasks[0].audio_path,
                            min_batch_size=1,
                            max_batch_size=self.max_batch_size,
                        )

                        if batch_size_to_use == 0:
                            logger.warning("Could not find suitable batch size. Falling back to non-batched mode.")
                            batched_model = None

                    if batched_model and batch_size_to_use > 0:
                        logger.info(f"Using batched inference with batch size: {batch_size_to_use}")

                except Exception as e:
                    logger.warning(f"Failed to setup batched inference: {str(e)}. Falling back to non-batched mode.")
                    batched_model = None

            for i, task in enumerate(tasks):
                logger.info(
                    _(
                        "info.processing",
                        task=self.generation_config["task"],
                        current=i + 1,
                        total=len(tasks),
                        path=task.audio_path,
                    )
                )

                use_batched = bool(
                    batched_model
                    and batch_size_to_use > 0
                    and self.vad_manager is not None
                    and self.generation_config.get("vad_filter", True)
                )
                if use_batched:
                    try:
                        segments, info, actual_batch_size = self._transcribe_batched(
                            model, batched_model, task, batch_size_to_use
                        )
                        if actual_batch_size < batch_size_to_use:
                            logger.info(f"Batch size auto-adjusted from {batch_size_to_use} to {actual_batch_size}")
                            batch_size_to_use = actual_batch_size
                    except Exception as e:
                        logger.warning(f"Batched inference failed: {str(e)}. Falling back to non-batched mode.")
                        segments, info = self._transcribe_smart_chunks(model, task)
                    duration_after_vad = info.duration_after_vad
                elif self._should_use_smart_split():
                    segments, info = self._transcribe_smart_chunks(model, task)
                    duration_after_vad = info.duration_after_vad
                else:
                    audio_input, transcription_config, manual_duration_after_vad = self._prepare_transcription(
                        task.audio_path
                    )
                    if manual_duration_after_vad == 0:
                        _segments = []
                        info = SimpleNamespace(
                            duration=len(audio_input) / WHISPER_SAMPLING_RATE,
                            duration_after_vad=0,
                        )
                    else:
                        _segments, info = model.transcribe(
                            audio_input,
                            **transcription_config,
                        )

                    duration_after_vad = (
                        manual_duration_after_vad if manual_duration_after_vad is not None else info.duration_after_vad
                    )

                    segments = []
                    for _segment in _segments:
                        segment = Segment(
                            start=int(_segment.start * 1_000),
                            end=int(_segment.end * 1_000),
                            text=_segment.text.strip(),
                        )
                        segments.append(segment)
                        logger.debug(
                            f"[{SubWriter.lrc_timestamp(segment.start)} --> "
                            f"{SubWriter.lrc_timestamp(segment.end)}] {segment.text}"
                        )

                self._log_duration(info.duration, duration_after_vad)

                speech_segments = getattr(info, "speech_segments", None)
                if speech_segments:
                    segments = split_segments_at_pauses(segments, speech_segments, self.pause_split_options)
                segments = filter_hallucinations(segments, self.hallucination_filter_options)
                segments = enforce_segment_timeline(segments, self.segment_merge_options.max_duration_ms)
                segments = merge_segments(segments, self.segment_merge_options)
                segments = enforce_segment_timeline(segments, self.segment_merge_options.max_duration_ms)
                os.makedirs(os.path.dirname(task.sub_prefix), exist_ok=True)
                for sub_suffix in task.sub_formats:
                    sub_path = f"{task.sub_prefix}.{sub_suffix}"
                    logger.info(_("info.writing", path=sub_path))
                    self.sub_writers[sub_suffix](segments, sub_path)

        finally:
            # Clean up VAD injection
            if self.vad_injected:
                uninject_vad()
                logger.info(_("info.vad_deactivated"))

    def _find_executable_batch_size(self, batched_model, sample_audio_path, min_batch_size=1, max_batch_size=64):
        """
        Find the maximum executable batch size for batched inference.
        Starts from max_batch_size and works down exponentially on OOM.

        Each attempt decodes exactly one full batch of 30s windows (taken from the sample file, repeated if
        it is too short) instead of transcribing the whole file.

        Args:
            batched_model: BatchedInferencePipeline instance
            sample_audio_path: Path to a sample audio file for testing
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try

        Returns:
            Optimal batch size that fits in memory
        """
        if not self.enable_batching:
            return 0

        logger.info(_("batch.finding_optimal", min_size=min_batch_size, max_size=max_batch_size))

        sample_audio = decode_audio(sample_audio_path, sampling_rate=WHISPER_SAMPLING_RATE)
        if len(sample_audio) == 0:
            sample_audio = np.zeros(WHISPER_SAMPLING_RATE, dtype=np.float32)
        config = self._batched_config()
        window_s = MAX_SMART_CHUNK_DURATION_S

        # Start from max and work down on failure (like HuggingFace Accelerate)
        current_batch_size = max_batch_size

        while current_batch_size >= min_batch_size:
            try:
                logger.info(_("batch.testing_size", size=current_batch_size))

                probe_audio = np.resize(sample_audio, int(current_batch_size * window_s * WHISPER_SAMPLING_RATE))
                clips = [{"start": i * window_s, "end": (i + 1) * window_s} for i in range(current_batch_size)]
                segments, info = batched_model.transcribe(
                    probe_audio,
                    batch_size=current_batch_size,
                    clip_timestamps=clips,
                    **config,
                )

                # Force evaluation by converting to list
                list(segments)

                # Success! This batch size works
                logger.info(_("batch.size_successful", size=current_batch_size))
                logger.info(_("batch.optimal_found", size=current_batch_size))
                return current_batch_size

            except RuntimeError as e:
                # If OOM, reduce batch size exponentially
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    logger.warning(_("batch.oom_error", size=current_batch_size))
                else:
                    logger.warning(
                        _(
                            "batch.runtime_error",
                            size=current_batch_size,
                            error=error_msg,
                        )
                    )

                # Reduce batch size by half (exponential backoff)
                new_batch_size = current_batch_size // 2

                # Ensure we reduce by at least 1
                if new_batch_size == current_batch_size:
                    new_batch_size = current_batch_size - 1

                if new_batch_size < min_batch_size:
                    logger.error(_("batch.no_suitable_size", min_size=min_batch_size))
                    return 0

                logger.info(
                    _(
                        "batch.reducing_size",
                        old_size=current_batch_size,
                        new_size=new_batch_size,
                    )
                )
                current_batch_size = new_batch_size

            except Exception as e:
                logger.warning(_("batch.unexpected_error", size=current_batch_size, error=str(e)))

                # Reduce batch size by half on unexpected errors too
                new_batch_size = current_batch_size // 2
                if new_batch_size < min_batch_size:
                    return 0
                current_batch_size = new_batch_size

        # Should not reach here
        logger.error(_("batch.no_suitable_size", min_size=min_batch_size))
        return 0

    def _transcribe_with_auto_batch_size(self, batched_model, audio_path, generation_config, starting_batch_size=None):
        """
        Transcribe with automatic batch size reduction on OOM.
        Similar to HuggingFace Accelerate's find_executable_batch_size decorator.

        This function automatically retries with smaller batch sizes if OOM occurs,
        implementing the same behavior as Accelerate's find_executable_batch_size.

        Args:
            batched_model: BatchedInferencePipeline instance
            audio_path: Path to audio file (or audio array)
            starting_batch_size: Initial batch size to try (uses self.batch_size if not specified)

        Returns:
            Tuple of (segments as a list, info, actual_batch_size_used)
        """
        batch_size = starting_batch_size or self.batch_size or 32
        min_batch_size = 1

        while batch_size >= min_batch_size:
            try:
                logger.debug(_("batch.attempting_transcription", size=batch_size))

                # Try transcription with current batch size. Decoding is lazy and that is where OOM
                # happens, so materialize the segments inside the try.
                segments_iter, info = batched_model.transcribe(audio_path, batch_size=batch_size, **generation_config)
                segments = list(segments_iter)

                # Success! Return results with the batch size that worked
                if batch_size < (starting_batch_size or self.batch_size or 32):
                    logger.info(_("batch.auto_adjusted", size=batch_size))

                return segments, info, batch_size

            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                    # Reduce batch size by 0.8 (20% reduction, similar to Accelerate's 0.9 but more aggressive)
                    new_batch_size = int(batch_size * 0.8)

                    # Ensure we reduce by at least 1
                    if new_batch_size == batch_size:
                        new_batch_size = batch_size - 1

                    logger.warning(
                        _(
                            "batch.oom_reducing",
                            old_size=batch_size,
                            new_size=new_batch_size,
                        )
                    )

                    batch_size = new_batch_size

                    if batch_size < min_batch_size:
                        logger.error(_("batch.cannot_run_min", min_size=min_batch_size))
                        raise RuntimeError(_("batch.inference_failed", min_size=min_batch_size)) from e
                else:
                    # Not an OOM error, re-raise
                    raise

        # Should not reach here
        raise RuntimeError("Failed to find executable batch size")

    def _scan(self, base_dirs) -> list[InferenceTask]:
        tasks: list[InferenceTask] = []

        def process(base_path, audio_path):
            nonlocal tasks
            p = Path(audio_path)
            suffix = p.suffix.lower().lstrip(".")

            logger.debug(_("debug.processing", path=audio_path))
            logger.debug(_("debug.file_suffix", suffix=suffix))
            logger.debug(_("debug.valid_suffixes", suffixes=self.audio_suffixes))

            if suffix not in self.audio_suffixes:
                logger.debug(_("debug.skipped_suffix", suffix=suffix))
                return

            rel_path = p.relative_to(base_path)
            abs_path = Path(os.path.join(self.output_dir or base_path, rel_path))
            sub_formats = []

            for suffix in self.sub_formats:
                sub_path = abs_path.parent / f"{abs_path.stem}.{suffix}"
                if sub_path.exists() and not self.overwrite:
                    logger.debug(_("debug.subtitle_exists", path=sub_path))
                    continue
                sub_formats.append(suffix)

            if len(sub_formats) == 0:
                logger.debug(_("debug.skipped_all_exist"))
                return

            logger.debug(_("debug.added_task", formats=sub_formats))
            tasks.append(InferenceTask(audio_path, str(abs_path.parent / abs_path.stem), sub_formats))

        for base_dir in base_dirs:
            # Expand user home directory
            base_dir = os.path.expanduser(base_dir)
            logger.debug(_("debug.scanning", path=base_dir))

            parent_dir = os.path.dirname(base_dir)
            if os.path.isdir(base_dir):
                for root, _dirs, files in os.walk(base_dir, topdown=True):
                    for file in files:
                        process(parent_dir, os.path.join(root, file))
            else:
                process(parent_dir, base_dir)

        logger.info(_("files.found", count=len(tasks)))
        return tasks


def diagnose_environment():
    """Run comprehensive environment diagnostics for debugging"""
    print("=" * 60)
    print("ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)

    # System info
    print("\n1. System Information:")
    print(f"   Platform: {platform.system()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Python: {sys.version}")
    print(f"   Executable: {sys.executable}")
    print(f"   Frozen: {getattr(sys, 'frozen', False)}")

    if getattr(sys, "frozen", False):
        print(f"   Bundle Dir: {getattr(sys, '_MEIPASS', 'Unknown')}")

    # CUDA environment
    print("\n2. CUDA Environment Variables:")
    cuda_vars = [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDNN_HOME",
        "LD_LIBRARY_PATH",
        "PATH",
    ]
    for var in cuda_vars:
        value = os.environ.get(var, "Not set")
        if var == "PATH" and value != "Not set":
            # Just show cuda-related paths
            cuda_paths = [p for p in value.split(os.pathsep) if "cuda" in p.lower() or "nvidia" in p.lower()]
            value = os.pathsep.join(cuda_paths) if cuda_paths else "No CUDA paths in PATH"
        print(f"   {var}: {value}")

    # Check for nvidia-smi
    print("\n3. NVIDIA GPU Detection:")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,cuda_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"   GPU Info: {result.stdout.strip()}")
        else:
            print("   nvidia-smi failed")
    except FileNotFoundError:
        print("   nvidia-smi not found in PATH")
    except Exception as e:
        print(f"   Error: {e}")


def check_onnxruntime_detailed():
    """Detailed ONNX Runtime check for debugging"""
    print("\n" + "=" * 60)
    print("ONNX RUNTIME DIAGNOSTICS")
    print("=" * 60)

    try:
        import onnxruntime as ort

        print("\n✓ onnxruntime imported successfully")
        print(f"  Version: {ort.__version__}")
        print(f"  Location: {ort.__file__}")

        # Check available providers
        providers = ort.get_available_providers()
        print(f"\n  Available providers: {providers}")

        # Check for GPU support
        has_cuda = "CUDAExecutionProvider" in providers
        has_tensorrt = "TensorrtExecutionProvider" in providers
        has_directml = "DmlExecutionProvider" in providers

        print("\n  GPU Support:")
        print(f"    CUDA: {'✓ Available' if has_cuda else '✗ Not Available'}")
        print(f"    TensorRT: {'✓ Available' if has_tensorrt else '✗ Not Available'}")
        print(f"    DirectML: {'✓ Available' if has_directml else '✗ Not Available'}")

        if not has_cuda and sys.platform != "darwin":
            print("\n  ⚠️ CUDA not available. This might be because:")
            print("    1. onnxruntime (CPU) is installed instead of onnxruntime-gpu")
            print("    2. CUDA libraries are missing or not in PATH")
            print("    3. Incompatible CUDA/cuDNN versions")

        # Check bundled libraries if frozen
        if getattr(sys, "frozen", False):
            bundle_dir = getattr(sys, "_MEIPASS", "")
            print(f"\n  Checking bundled libraries in: {bundle_dir}")

            cuda_libs = []
            onnx_libs = []

            try:
                for _root, _dirs, files in os.walk(bundle_dir):
                    for file in files:
                        if any(x in file.lower() for x in ["cuda", "cudnn", "cublas", "cufft"]):
                            cuda_libs.append(file)
                        elif "onnx" in file.lower():
                            onnx_libs.append(file)

                if cuda_libs:
                    print(f"\n  Found {len(cuda_libs)} CUDA-related libraries:")
                    for lib in cuda_libs[:10]:
                        print(f"    - {lib}")
                    if len(cuda_libs) > 10:
                        print(f"    ... and {len(cuda_libs) - 10} more")
                else:
                    print("\n  ⚠️ No CUDA libraries found in bundle")
            except Exception as e:
                print(f"  Error scanning bundle: {e}")

        return True

    except ImportError as e:
        print(f"\n✗ Failed to import onnxruntime: {e}")
        print("\nSuggestions:")
        print("  1. Install onnxruntime-gpu for GPU support")
        print("  2. Check if package is bundled correctly in PyInstaller")
        return False
    except Exception as e:
        print(f"\n✗ Error during ONNX Runtime check: {e}")
        traceback.print_exc()
        return False


def test_vad_initialization():
    """Test VAD model initialization for debugging"""
    print("\n" + "=" * 60)
    print("VAD MODEL TEST")
    print("=" * 60)

    try:
        from .vad_manager import VadModelManager, WhisperVADOnnxWrapper  # noqa: F401

        print("✓ VAD modules imported successfully")

        # Check for model files
        model_paths = [
            "models/whisper_vad.onnx",
            "models/vad/whisper_vad.onnx",
            os.path.join(os.path.dirname(sys.executable), "models", "whisper_vad.onnx"),
        ]

        # If frozen, also check in bundle directory
        if getattr(sys, "frozen", False):
            bundle_dir = getattr(sys, "_MEIPASS", "")
            model_paths.extend(
                [
                    os.path.join(bundle_dir, "models", "whisper_vad.onnx"),
                    os.path.join(bundle_dir, "whisper_vad.onnx"),
                ]
            )

        model_path = None
        print("\nSearching for VAD model:")
        for path in model_paths:
            exists = os.path.exists(path)
            print(f"  {path}: {'Found' if exists else 'Not found'}")
            if exists and model_path is None:
                model_path = path

        if model_path:
            print(f"\n✓ Using model: {model_path}")

            # Try to initialize
            print("\nTesting VAD initialization (GPU if available):")
            try:
                wrapper = WhisperVADOnnxWrapper(model_path=model_path, force_cpu=False, num_threads=1)
                print(f"  ✓ Device: {wrapper.device}")
                print(f"  ✓ Providers: {wrapper.session.get_providers()}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

            # Test with forced CPU for comparison
            print("\nTesting VAD initialization (Force CPU):")
            try:
                wrapper_cpu = WhisperVADOnnxWrapper(model_path=model_path, force_cpu=True, num_threads=1)
                print(f"  ✓ Device: {wrapper_cpu.device}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print("\n✗ No VAD model file found")
            print("  Download the model using download_models.py")

    except ImportError as e:
        print(f"✗ Failed to import VAD modules: {e}")
    except Exception as e:
        print(f"✗ Error during VAD test: {e}")
        traceback.print_exc()


def launch_debug_console():
    """Launch interactive Python console for debugging"""
    print("\n" + "=" * 60)
    print("INTERACTIVE DEBUG CONSOLE")
    print("=" * 60)
    print("\nYou now have access to an interactive Python console.")
    print("\nAvailable commands:")
    print("  diagnose()       - Run environment diagnostics")
    print("  check_onnx()     - Check ONNX Runtime status")
    print("  test_vad()       - Test VAD initialization")
    print("  import X         - Try importing any module")
    print("  exit() or Ctrl+D - Exit console and continue")
    print("\nUseful variables:")
    print("  sys.path         - Python module search paths")
    print("  os.environ       - Environment variables")
    print("  sys.frozen       - Check if running from PyInstaller")
    print("=" * 60 + "\n")

    # Create namespace with useful functions
    namespace = {
        "diagnose": diagnose_environment,
        "check_onnx": check_onnxruntime_detailed,
        "test_vad": test_vad_initialization,
        "sys": sys,
        "os": os,
        "platform": platform,
    }

    # Launch interactive console
    code.InteractiveConsole(locals=namespace).interact(banner="")


def main():
    """Main entry point for the script"""
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        # When run as a module, don't change directory
        pass

    args = parse_arguments()

    # fix windows gbk pipe issue
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    # Display open-source notice
    print("=" * 70)
    print("⚠️  重要声明 / IMPORTANT NOTICE")
    print("=" * 70)
    print("本软件开源于: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice")
    print("开发团队: AI汉化组 (https://t.me/transWithAI)")
    print("任何第三方非免费下载均为智商税")
    print("=" * 70)
    print()

    # Check if console mode requested
    if args.console:
        # Run diagnostics first
        diagnose_environment()
        check_onnxruntime_detailed()
        test_vad_initialization()

        # Launch interactive console
        launch_debug_console()

        # After console exits, ask if user wants to continue with normal operation
        print("\nDebug console exited.")
        try:
            response = input("Continue with normal inference? (y/n): ").strip().lower()
            if response != "y":
                print("Exiting...")
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

    # Normal operation
    # Set logger to DEBUG so file handler captures everything;
    # console handler stays at the user-requested level.
    logger.setLevel(logging.DEBUG)
    log_handler.setLevel(args.log_level)

    # Add file logging to latest.log in current working directory
    # This helps users report issues by providing a log file
    log_file_path = os.path.join(os.getcwd(), "latest.log")
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    file_handler.setLevel(logging.DEBUG)

    # Add file handler to the module logger
    logger.addHandler(file_handler)

    logger.info(_("info.logging_to_file").format(path=log_file_path))
    logger.info(_("info.program_version").format(version="v1.10"))
    logger.info(_("info.python_version").format(version=sys.version))
    logger.info(_("info.platform").format(platform=platform.platform()))
    logger.info(_("info.arguments").format(args=vars(args)))

    if len(args.base_dirs) == 0:
        logger.warning(_("warnings.drag_files"))
        sys.exit(1)

    inference = Inference(args)
    inference.generates(args.base_dirs)
    sys.exit(0)


if __name__ == "__main__":
    # When run directly as a script
    import os

    os.chdir(os.path.dirname(__file__))
    main()
