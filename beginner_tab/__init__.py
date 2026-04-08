# ==========================================================================
# File:        __init__.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Package initializer for beginner_tab. Re-exports every public
#              class and exception from the sub-modules so that callers can
#              write ``from beginner_tab import AudioLoader`` instead of
#              reaching into individual files.
# ==========================================================================

"""beginner_tab — Convert audio to beginner-friendly guitar tabs.

This package provides a complete pipeline for transforming an audio file
into an ASCII guitar tablature suitable for beginners:

    AudioLoader        → load and normalize audio (MP3/WAV/M4A)
    PitchTracker       → monophonic pitch detection via pYIN
    BasicPitchTracker  → polyphonic pitch detection via Spotify Basic-pitch
    TabSimplifier      → quantize raw events onto a beat grid
    FretboardMapper    → map MIDI notes to guitar string/fret positions
    TabRenderer        → render the result as a printable ASCII tab
    TabStorage         → persist generated tabs in a SQLite database
"""

from .audio_loader import AudioLoader, AudioLoadError
from .pitch_tracker import PitchTracker, BasicPitchTracker, PitchDetectionError
from .tab_simplifier import TabSimplifier
from .fretboard_mapper import FretboardMapper
from .tab_renderer import TabRenderer
from .tab_storage import TabStorage, TabStorageError

# ---------------------------------------------------------------------------
# Probe whether the Basic-pitch neural network can actually run.
# Just importing basic_pitch is not enough — the ONNX (or TF) model file
# must also exist on disk.  On Streamlit Cloud basic-pitch is absent from
# requirements.txt (tensorflow incompatible).  Locally, the user may be
# running the wrong Python interpreter (system Python instead of venv).
# The probe stores a human-readable reason so the UI can show a helpful
# message instead of a cryptic error.
# ---------------------------------------------------------------------------
def _probe_basic_pitch() -> tuple[bool, str]:
    """Return (available, reason) for the Basic-pitch detector.

    Returns:
        Tuple of (True, "") if basic-pitch is ready, or
        (False, "<human-readable reason>") explaining why it is not.
    """
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        return False, (
            "basic-pitch is not installed in this Python environment. "
            "Activate the project venv and install dependencies:\n\n"
            "```\nsource venv/bin/activate\n"
            "pip install -r requirements-local.txt\n```"
        )
    except Exception as exc:
        return False, f"basic-pitch import failed: {exc}"

    import pathlib
    model_path = pathlib.Path(ICASSP_2022_MODEL_PATH)
    onnx_path = model_path.with_suffix(".onnx")
    if onnx_path.exists() or model_path.exists():
        return True, ""
    return False, (
        f"basic-pitch is installed but the model file is missing.\n"
        f"Expected: `{onnx_path}` or `{model_path}`"
    )

BASIC_PITCH_AVAILABLE: bool
BASIC_PITCH_UNAVAILABLE_REASON: str
BASIC_PITCH_AVAILABLE, BASIC_PITCH_UNAVAILABLE_REASON = _probe_basic_pitch()

# Symbols available when a consumer does ``from beginner_tab import *``
__all__ = [
    "AudioLoader",
    "AudioLoadError",
    "PitchTracker",
    "BasicPitchTracker",
    "PitchDetectionError",
    "TabSimplifier",
    "FretboardMapper",
    "TabRenderer",
    "TabStorage",
    "TabStorageError",
    "BASIC_PITCH_AVAILABLE",
    "BASIC_PITCH_UNAVAILABLE_REASON",
]
__version__ = "0.2.0"
