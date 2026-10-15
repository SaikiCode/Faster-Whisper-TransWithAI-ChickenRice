"""
VAD Model Manager - Manages different VAD model implementations
"""

import json
import logging
import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# Import modern i18n module for translations
from . import i18n_modern as i18n

# Convenience imports
_ = i18n._

logger = logging.getLogger(__name__)


@dataclass
class VadConfig:
    """Configuration for VAD models"""

    default_model: str = "whisper_vad"
    auto_inject: bool = False
    ttl: int = 3600  # Cache TTL in seconds

    # VAD parameters
    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400

    # ONNX-specific parameters
    onnx_model_path: str | None = None
    onnx_metadata_path: str | None = None
    whisper_model_name: str = "openai/whisper-base"
    frame_duration_ms: int = 20
    chunk_duration_ms: int = 30000
    force_cpu: bool = False
    num_threads: int = 1


class VadModel(Protocol):
    """Protocol for VAD models"""

    @property
    def frame_duration_ms(self) -> int:
        """Duration of one probability frame"""
        ...

    def get_speech_probs(self, audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        """Get frame-level speech probabilities from audio"""
        ...

    def get_speech_timestamps(self, audio: np.ndarray, sampling_rate: int = 16000, **kwargs) -> list[dict[str, Any]]:
        """Get speech timestamps from audio"""
        ...


class WhisperVADOnnxWrapper:
    """ONNX wrapper for Whisper-based VAD model following Silero's architecture."""

    def __init__(
        self,
        model_path: str,
        metadata_path: str | None = None,
        force_cpu: bool = False,
        num_threads: int = 1,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ):
        """Initialize ONNX model wrapper.

        Args:
            model_path: Path to ONNX model file
            metadata_path: Path to metadata JSON file (optional)
            force_cpu: Force CPU execution even if GPU is available
            num_threads: Number of CPU threads for inference
            progress_callback: Optional callback for progress tracking (chunk_idx, total_chunks, device)
        """
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(_("vad.onnx_not_installed")) from err

        try:
            from transformers import WhisperFeatureExtractor
        except ImportError as err:
            raise ImportError(_("vad.transformers_not_installed")) from err

        self.model_path = model_path
        self.progress_callback = progress_callback
        self.device = "CPU"  # Will be updated based on actual provider

        # Load metadata
        if metadata_path is None:
            metadata_path = model_path.replace(".onnx", "_metadata.json")

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            warnings.warn("No metadata file found. Using default values.")
            self.metadata = {
                "whisper_model_name": "openai/whisper-base",
                "frame_duration_ms": 20,
                "total_duration_ms": 30000,
            }

        # Initialize feature extractor - try local folder first for offline usage
        local_whisper_base_path = Path("models/whisper-base")
        if local_whisper_base_path.exists() and (local_whisper_base_path / "preprocessor_config.json").exists():
            # Load from local folder for offline usage
            try:
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(str(local_whisper_base_path))
                logger.info(_("vad.feature_extractor_loaded", path=local_whisper_base_path))
            except Exception as e:
                warnings.warn(f"Failed to load from local folder, trying online: {e}")
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.metadata["whisper_model_name"])
        else:
            # Try to load from HuggingFace (requires internet)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.metadata["whisper_model_name"])

        # Set up ONNX Runtime session
        opts = ort.SessionOptions()
        # The graph runs sequentially, so inter-op threads are never used; parallelism comes from intra-op threads.
        opts.inter_op_num_threads = 1

        # Determine execution provider first
        providers = ["CPUExecutionProvider"]
        use_gpu = not force_cpu and "CUDAExecutionProvider" in ort.get_available_providers()

        if use_gpu:
            providers.insert(0, "CUDAExecutionProvider")
            self.device = "GPU (CUDA)"
            # For GPU, use the provided num_threads or default
            opts.intra_op_num_threads = num_threads
        else:
            self.device = "CPU"
            # For CPU, use half of available processors if num_threads is default (1)
            import multiprocessing

            if num_threads == 1:
                # Use half of CPU count for optimal performance
                optimal_threads = max(1, multiprocessing.cpu_count() // 2)
                opts.intra_op_num_threads = optimal_threads
                logger.info(_("vad.auto_configured", threads=optimal_threads, total=multiprocessing.cpu_count()))
            else:
                # Use user-specified thread count
                opts.intra_op_num_threads = num_threads

        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=opts)

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Model parameters
        self.sample_rate = 16000  # Whisper uses 16kHz
        self.frame_duration_ms = self.metadata.get("frame_duration_ms", 20)
        self.chunk_duration_ms = self.metadata.get("total_duration_ms", 30000)
        self.chunk_samples = int(self.chunk_duration_ms * self.sample_rate / 1000)
        self.frames_per_chunk = int(self.chunk_duration_ms / self.frame_duration_ms)

        # Initialize state
        self.reset_states()

        logger.info(_("vad.model_loaded", path=model_path))
        logger.info(_("vad.device", device=self.device))
        logger.info(_("vad.providers", providers=providers))
        logger.info(_("vad.chunk_duration", duration=self.chunk_duration_ms))
        logger.info(_("vad.frame_duration", duration=self.frame_duration_ms))

    def reset_states(self):
        """Reset internal states for new audio stream."""
        self._context = None
        self._last_chunk = None

    def _validate_input(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Validate and preprocess input audio.

        Args:
            audio: Input audio array
            sr: Sample rate

        Returns:
            Preprocessed audio at 16kHz
        """
        if audio.ndim > 1:
            # Convert to mono if multi-channel
            audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)

        # Resample if needed
        if sr != self.sample_rate:
            try:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            except ImportError:
                logger.warning(_("vad.librosa_not_installed"))
                # Basic downsampling if librosa not available
                if sr > self.sample_rate:
                    # Simple downsampling
                    ratio = sr // self.sample_rate
                    audio = audio[::ratio]

        return audio

    def __call__(self, audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Process a single audio chunk.

        Args:
            audio_chunk: Audio chunk to process
            sr: Sample rate

        Returns:
            Frame-level speech probabilities
        """
        # Validate input
        audio_chunk = self._validate_input(audio_chunk, sr)

        # Ensure chunk is correct size
        if len(audio_chunk) < self.chunk_samples:
            audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)), mode="constant")
        elif len(audio_chunk) > self.chunk_samples:
            audio_chunk = audio_chunk[: self.chunk_samples]

        # Extract features
        inputs = self.feature_extractor(audio_chunk, sampling_rate=self.sample_rate, return_tensors="np")

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: inputs.input_features})
        if self.device != "CPU" and "CUDAExecutionProvider" not in self.session.get_providers():
            # ONNX Runtime fell back to CPU inside run() (e.g. a cuDNN library failed to load).
            self.device = "CPU"
            logger.warning(_("vad.fell_back_to_cpu"))

        # Apply sigmoid to get probabilities
        frame_logits = outputs[0][0]  # Remove batch dimension
        frame_probs = 1 / (1 + np.exp(-frame_logits))

        return frame_probs

    def audio_forward(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Process full audio file in chunks (Silero-style).

        Args:
            audio: Full audio array
            sr: Sample rate

        Returns:
            Concatenated frame probabilities for entire audio
        """
        audio = self._validate_input(audio, sr)
        self.reset_states()

        all_probs = []

        # Calculate total number of chunks
        total_chunks = (len(audio) + self.chunk_samples - 1) // self.chunk_samples

        # Log initial processing info
        logger.info(_("vad.starting", device=self.device))
        logger.info(_("vad.total_samples", samples=len(audio)))
        logger.info(_("vad.chunk_size", samples=self.chunk_samples, duration=self.chunk_duration_ms))
        logger.info(_("vad.total_chunks", chunks=total_chunks))

        # Process in chunks
        for chunk_idx, i in enumerate(range(0, len(audio), self.chunk_samples)):
            chunk = audio[i : i + self.chunk_samples]

            # Pad last chunk if needed
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode="constant")

            # Report progress
            if self.progress_callback:
                self.progress_callback(chunk_idx + 1, total_chunks, self.device)

            # Log chunk progress
            progress_pct = ((chunk_idx + 1) / total_chunks) * 100
            logger.debug(
                _(
                    "vad.processing_chunk",
                    current=chunk_idx + 1,
                    total=total_chunks,
                    percent=progress_pct,
                    device=self.device,
                )
            )

            # Get predictions for chunk
            chunk_probs = self.__call__(chunk, self.sample_rate)
            all_probs.append(chunk_probs)

        logger.info(_("vad.completed", chunks=total_chunks, device=self.device))

        # Concatenate all probabilities
        if all_probs:
            return np.concatenate(all_probs)
        return np.array([])


def get_speech_timestamps_onnx(
    audio: np.ndarray,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = True,
    neg_threshold: float | None = None,
    progress_tracking_callback: Callable[[float], None] | None = None,
) -> list[dict[str, float]]:
    """Extract speech timestamps from audio using Silero-style processing.

    This function implements Silero VAD's approach with:
    - Dual threshold (positive and negative) for hysteresis
    - Proper segment padding
    - Minimum duration filtering
    - Maximum duration handling with intelligent splitting

    Args:
        audio: Input audio array
        model: VAD model (WhisperVADOnnxWrapper instance)
        threshold: Speech threshold (default: 0.5)
        sampling_rate: Audio sample rate
        min_speech_duration_ms: Minimum speech segment duration
        max_speech_duration_s: Maximum speech segment duration
        min_silence_duration_ms: Minimum silence to split segments
        speech_pad_ms: Padding to add to speech segments
        return_seconds: Return times in seconds vs samples
        neg_threshold: Negative threshold for hysteresis (default: threshold - 0.15)
        progress_tracking_callback: Progress callback function

    Returns:
        List of speech segments with start/end times
    """
    # Audio should already be numpy array

    # Validate audio
    if audio.ndim > 1:
        audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)

    # Get frame probabilities for entire audio
    model.reset_states()
    speech_probs = model.audio_forward(audio, sampling_rate)

    return speech_probs_to_timestamps(
        speech_probs,
        frame_duration_ms=model.frame_duration_ms,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=return_seconds,
        neg_threshold=neg_threshold,
        progress_tracking_callback=progress_tracking_callback,
    )


def resolve_neg_threshold(threshold: float, neg_threshold: float | None) -> float:
    """Negative (speech offset) threshold for hysteresis; defaults to threshold - 0.15."""
    if neg_threshold is None:
        return max(threshold - 0.15, 0.01)
    return neg_threshold


def speech_probs_to_timestamps(
    speech_probs: np.ndarray,
    *,
    frame_duration_ms: int = 20,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = True,
    neg_threshold: float | None = None,
    progress_tracking_callback: Callable[[float], None] | None = None,
) -> list[dict[str, float]]:
    """Turn frame-level speech probabilities into speech segments (Silero-style post-processing).

    Split from get_speech_timestamps_onnx so probabilities computed once for a whole file can be
    reused for any sub-range of it without running the ONNX model again.
    """
    # Calculate frame parameters
    frame_samples = int(sampling_rate * frame_duration_ms / 1000)

    # Convert durations to frames
    min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
    min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
    speech_pad_frames = int(speech_pad_ms / frame_duration_ms)
    max_speech_frames = (
        int(max_speech_duration_s * 1000 / frame_duration_ms)
        if max_speech_duration_s != float("inf")
        else len(speech_probs)
    )

    # Set negative threshold for hysteresis
    neg_threshold = resolve_neg_threshold(threshold, neg_threshold)

    # Track speech segments
    triggered = False
    speeches: list[dict[str, float]] = []
    current_speech: dict[str, float] = {}
    temp_end = 0

    def close_segment(end: int) -> None:
        start = int(current_speech["start"])
        current_speech["end"] = end
        if end > start:
            current_speech["probability"] = float(np.mean(speech_probs[start:end]))
        speeches.append(current_speech)

    # Process each frame
    for i, speech_prob in enumerate(speech_probs):
        # Report progress
        if progress_tracking_callback:
            progress = (i + 1) / len(speech_probs) * 100
            progress_tracking_callback(progress)

        # Speech onset detection
        if speech_prob >= threshold and not triggered:
            triggered = True
            current_speech["start"] = i
            continue

        # Check for maximum speech duration
        if triggered and i - current_speech["start"] > max_speech_frames:
            # Split at the least speech-like frame in the second half of the allowed span instead of
            # cutting at the exact limit, and keep the speech going from there (no frames dropped).
            start = int(current_speech["start"])
            window_start = start + max(1, max_speech_frames // 2)
            split = window_start + int(np.argmin(speech_probs[window_start : start + max_speech_frames + 1]))
            close_segment(split)
            current_speech = {"start": split}
            if temp_end and temp_end <= split:
                temp_end = 0

        # Speech offset detection with hysteresis
        if speech_prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = i

            # Check if silence is long enough
            if i - temp_end >= min_silence_frames:
                # End current speech segment if it is long enough
                if temp_end - current_speech["start"] >= min_speech_frames:
                    close_segment(temp_end)

                current_speech = {}
                triggered = False
                temp_end = 0

        # Reset temp_end if speech resumes
        elif speech_prob >= threshold and temp_end:
            temp_end = 0

    # Handle speech that continues to the end
    if triggered and len(speech_probs) - current_speech["start"] >= min_speech_frames:
        close_segment(len(speech_probs))

    # Apply padding to segments
    for i, speech in enumerate(speeches):
        # Add padding
        if i == 0:
            speech["start"] = max(0, speech["start"] - speech_pad_frames)
        else:
            speech["start"] = max(speeches[i - 1]["end"], speech["start"] - speech_pad_frames)

        if i < len(speeches) - 1:
            speech["end"] = min(speeches[i + 1]["start"], speech["end"] + speech_pad_frames)
        else:
            speech["end"] = min(len(speech_probs), speech["end"] + speech_pad_frames)

    # Convert to time units or sample indices based on return_seconds
    for speech in speeches:
        if return_seconds:
            # Convert frame indices to seconds
            speech["start"] = speech["start"] * frame_duration_ms / 1000
            speech["end"] = speech["end"] * frame_duration_ms / 1000
        else:
            # Convert frame indices to sample indices
            speech["start"] = int(speech["start"] * frame_samples)
            speech["end"] = int(speech["end"] * frame_samples)

    return speeches


class WhisperVadModel:
    """
    Whisper-based VAD model implementation using ONNX.
    Uses a Whisper model exported to ONNX for voice activity detection.
    """

    def __init__(
        self, config: VadConfig | None = None, progress_callback: Callable[[int, int, str], None] | None = None
    ):
        self.config = config or VadConfig()
        self.wrapper = None
        self.progress_callback = progress_callback

        # Initialize ONNX model if path provided
        if self.config.onnx_model_path and os.path.exists(self.config.onnx_model_path):
            try:
                self.wrapper = WhisperVADOnnxWrapper(
                    model_path=self.config.onnx_model_path,
                    metadata_path=self.config.onnx_metadata_path,
                    force_cpu=self.config.force_cpu,
                    num_threads=self.config.num_threads,
                    progress_callback=progress_callback,
                )
                logger.info(_("vad.model_initialized", path=self.config.onnx_model_path))
                if self.wrapper.device:
                    logger.info(_("vad.using_device", device=self.wrapper.device))
            except Exception as e:
                logger.error(_("vad.init_failed", error=e))
        else:
            logger.warning(_("vad.path_invalid", path=self.config.onnx_model_path))

    @property
    def frame_duration_ms(self) -> int:
        if self.wrapper is not None:
            return int(self.wrapper.frame_duration_ms)
        return int(self.config.frame_duration_ms)

    def get_speech_probs(self, audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        """Frame-level speech probabilities, trimmed to the frames that actually cover the audio."""
        if self.wrapper is None:
            logger.error(_("vad.not_initialized"))
            return np.zeros(0, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)

        self.wrapper.reset_states()
        speech_probs = self.wrapper.audio_forward(audio, sampling_rate)
        # audio_forward pads the last 30s window with zeros; drop the frames past the end of the audio.
        resampled_samples = len(audio) * self.wrapper.sample_rate / sampling_rate
        frame_samples = self.wrapper.sample_rate * self.frame_duration_ms / 1000
        return speech_probs[: int(np.ceil(resampled_samples / frame_samples))]

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        threshold: float | None = None,
        min_speech_duration_ms: int | None = None,
        max_speech_duration_s: float | None = None,
        min_silence_duration_ms: int | None = None,
        speech_pad_ms: int | None = None,
        neg_threshold: float | None = None,
        speech_probs: np.ndarray | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Get speech timestamps using Whisper VAD.

        If speech_probs is given (precomputed with get_speech_probs), the ONNX model is not run.
        """
        if speech_probs is None:
            if self.wrapper is None:
                logger.error(_("vad.not_initialized"))
                return []
            speech_probs = self.get_speech_probs(audio, sampling_rate)

        # Use provided parameters or defaults from config
        threshold = threshold if threshold is not None else self.config.threshold
        neg_threshold = neg_threshold if neg_threshold is not None else self.config.neg_threshold
        min_speech_duration_ms = (
            min_speech_duration_ms if min_speech_duration_ms is not None else self.config.min_speech_duration_ms
        )
        max_speech_duration_s = (
            max_speech_duration_s if max_speech_duration_s is not None else self.config.max_speech_duration_s
        )
        min_silence_duration_ms = (
            min_silence_duration_ms if min_silence_duration_ms is not None else self.config.min_silence_duration_ms
        )
        speech_pad_ms = speech_pad_ms if speech_pad_ms is not None else self.config.speech_pad_ms

        # Return sample indices (not seconds) for compatibility with faster_whisper
        segments = speech_probs_to_timestamps(
            speech_probs,
            frame_duration_ms=self.frame_duration_ms,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_s=max_speech_duration_s,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False,  # faster_whisper expects sample indices
            neg_threshold=neg_threshold,
        )

        logger.debug(_("vad.speech_segments", count=len(segments)))
        return segments

    def get_device(self) -> str:
        """Get the device being used for VAD processing."""
        if self.wrapper:
            return self.wrapper.device
        return "Not initialized"


class VadModelManager:
    """
    Manages different VAD model implementations.
    Provides a unified interface for VAD operations.
    """

    def __init__(
        self,
        config: VadConfig | None = None,
        ttl: int = 3600,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ):
        self.config = config or VadConfig()
        self.ttl = ttl
        self.progress_callback = progress_callback
        self._models: dict[str, VadModel] = {}  # Instance variable, not class variable
        self._config = config

        # Register available models
        self._register_models()

    def _register_models(self):
        """Register available VAD models"""
        # Always recreate the model with the current progress callback
        self._models["whisper_vad"] = WhisperVadModel(self.config, progress_callback=self.progress_callback)
        logger.debug(_("vad.registered"))

    def get_model(self, model_id: str) -> VadModel:
        """Get a VAD model by ID"""
        if model_id not in self._models:
            logger.warning(_("vad.model_not_found", model_id=model_id))
            model_id = self.config.default_model

        return self._models.get(model_id, self._models["whisper_vad"])

    def get_speech_timestamps(
        self, model_id: str, audio: np.ndarray, sampling_rate: int = 16000, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Get speech timestamps using specified model.

        Args:
            model_id: ID of the VAD model to use
            audio: Audio array
            sampling_rate: Sample rate of audio
            **kwargs: Additional parameters for the VAD model

        Returns:
            List of speech segments with start, end, and probability
        """
        model = self.get_model(model_id)
        return model.get_speech_timestamps(audio, sampling_rate, **kwargs)

    def get_speech_probs(self, model_id: str, audio: np.ndarray, sampling_rate: int = 16000) -> np.ndarray:
        """Get frame-level speech probabilities (reusable via get_speech_timestamps(speech_probs=...))."""
        return self.get_model(model_id).get_speech_probs(audio, sampling_rate)

    @property
    def frame_duration_ms(self) -> int:
        """Duration of one speech probability frame of the default model."""
        return self.get_model(self.config.default_model).frame_duration_ms

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of available VAD models"""
        # Since models are now instance variables, return the known model types
        return ["whisper_vad"]

    def get_device(self, model_id: str | None = None) -> str:
        """Get the device being used for VAD processing."""
        if model_id is None:
            model_id = self.config.default_model
        model = self.get_model(model_id)
        if hasattr(model, "get_device"):
            return model.get_device()
        return "Unknown"
