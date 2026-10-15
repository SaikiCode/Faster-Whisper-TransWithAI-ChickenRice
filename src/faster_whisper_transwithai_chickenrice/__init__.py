"""
faster_whisper_transwithai_chickenrice - Custom VAD injection for faster_whisper
"""

from .injection import (
    VadInjectionContext,
    VadOptionsCompat,
    auto_inject_vad,
    inject_vad,
    is_injection_active,
    precomputed_speech_probs,
    uninject_vad,
    with_vad_injection,
)
from .vad_manager import VadModelManager, WhisperVadModel

__version__ = "0.1.0"

__all__ = [
    "inject_vad",
    "uninject_vad",
    "VadInjectionContext",
    "with_vad_injection",
    "auto_inject_vad",
    "VadOptionsCompat",
    "is_injection_active",
    "precomputed_speech_probs",
    "VadModelManager",
    "WhisperVadModel",
]
