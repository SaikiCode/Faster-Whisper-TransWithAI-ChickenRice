#!/usr/bin/env python3
"""
Runtime hook for PyInstaller to set environment variables before the application starts.
This resolves OpenMP conflicts when multiple libraries bring their own OpenMP implementations.
"""

import glob
import multiprocessing
import os
import sys

# Set KMP_DUPLICATE_LIB_OK to allow multiple OpenMP libraries
# This is needed because different packages (numpy, scipy, ctranslate2, onnxruntime)
# may bring different OpenMP implementations (libiomp5md.dll vs mk2iomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress transformers advisory warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Configure ONNX Runtime to use half of available CPU cores for better performance
# This prevents oversubscription and resource contention
cpu_count = multiprocessing.cpu_count()
optimal_threads = max(1, cpu_count // 2)

# Set ONNX Runtime environment variables for CPU execution
os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
os.environ["MKL_NUM_THREADS"] = str(optimal_threads)


def _add_windows_dll_directory(path: str) -> None:
    """Best-effort DLL search path setup (Windows only)."""
    if not path or not os.path.isdir(path):
        return
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return
    try:
        add_dir(path)
        print(f"Runtime hook: Added DLL directory: {path}")
    except Exception as e:
        print(f"Runtime hook: Failed to add DLL directory '{path}': {e}")


def _disable_error_dialogs() -> None:
    """
    Make a failing DLL load return an error instead of showing a modal "entry point not found" dialog.

    The dialog blocks the process until someone clicks OK, which hangs unattended runs (e.g. a service
    launching infer.exe); without it the library reports the error and the caller can fall back.
    """
    import ctypes

    sem_failcriticalerrors = 0x0001
    sem_noopenfileerrorbox = 0x8000
    kernel32 = ctypes.windll.kernel32
    kernel32.SetErrorMode(kernel32.SetErrorMode(0) | sem_failcriticalerrors | sem_noopenfileerrorbox)
    print("Runtime hook: Disabled DLL load error dialogs")


def _prefer_bundled_cudnn(bundle_dir: str) -> None:
    """
    Put the directories of the bundled cuDNN at the front of PATH.

    cuDNN 9 loads its sub-libraries (cudnn_engines_precompiled64_9.dll, ...) by name through the standard
    DLL search, which covers PATH but not the bundle directory. A CUDA toolkit on PATH that ships another
    cuDNN 9 build then supplies a mismatched sub-library and the GPU VAD falls back to the CPU.
    """
    patterns = ("cudnn64_*.dll", "*/cudnn64_*.dll", "*/*/cudnn64_*.dll", "*/*/*/cudnn64_*.dll")
    cudnn_dirs: list[str] = []
    for pattern in patterns:
        for dll in sorted(glob.glob(os.path.join(bundle_dir, pattern))):
            directory = os.path.dirname(dll)
            if directory not in cudnn_dirs:
                cudnn_dirs.append(directory)
    if not cudnn_dirs:
        return
    os.environ["PATH"] = os.pathsep.join([*cudnn_dirs, os.environ.get("PATH", "")])
    for directory in cudnn_dirs:
        _add_windows_dll_directory(directory)
    print(f"Runtime hook: Put bundled cuDNN first on PATH: {cudnn_dirs}")


if sys.platform == "win32" and getattr(sys, "frozen", False):
    _disable_error_dialogs()
    _prefer_bundled_cudnn(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))

    # If this build bundles AMD ROCm runtime wheels, ensure their DLL directories are
    # on the search path before importing GPU libraries (e.g., ctranslate2).
    bundle_root = os.path.dirname(sys.executable)
    _add_windows_dll_directory(os.path.join(bundle_root, "_rocm_sdk_core", "bin"))
    _add_windows_dll_directory(os.path.join(bundle_root, "_rocm_sdk_libraries_custom", "bin"))

print("Runtime hook: Set KMP_DUPLICATE_LIB_OK=TRUE to resolve OpenMP conflicts")
print("Runtime hook: Set TRANSFORMERS_NO_ADVISORY_WARNINGS=1 to suppress advisory warnings")
print(f"Runtime hook: Configured ONNX Runtime to use {optimal_threads} threads (half of {cpu_count} available CPUs)")
