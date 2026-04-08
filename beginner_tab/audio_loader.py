# ==========================================================================
# File:        audio_loader.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Loads audio files (MP3, WAV, M4A, OGG, FLAC), converts
#              non-WAV formats to a temporary WAV via pydub + ffmpeg,
#              and returns a normalised mono float32 NumPy array at
#              22 050 Hz for downstream pitch analysis.
# ==========================================================================

"""Audio loading and WAV-conversion module.

Supports MP3, WAV, and M4A input.  Non-WAV files are converted to a
temporary WAV via pydub, which requires the **ffmpeg** binary to be
present on the system PATH.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

# Set of file extensions that AudioLoader will accept
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac"})


class AudioLoadError(Exception):
    """Raised when an audio file cannot be loaded or decoded."""


class AudioLoader:
    """Load and normalise an audio file for downstream pitch analysis.

    Converts any supported format to mono, 22 050 Hz WAV in memory.
    This sample rate is the librosa default and works well with pYIN.

    Example::

        loader = AudioLoader("song.mp3")
        audio, sr = loader.load()
    """

    TARGET_SR: int = 22_050  # Hz — standard librosa default

    def __init__(self, file_path: str) -> None:
        """Initialise with the path to an audio file.

        Args:
            file_path: Absolute or relative path to the source audio.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ValueError: If the file extension is not in SUPPORTED_EXTENSIONS.
        """
        self._path = Path(file_path)

        if not self._path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if self._path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{self._path.suffix}'.  "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        self._temp_wav: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> tuple[np.ndarray, int]:
        """Load the audio file and return a mono float32 array.

        Returns:
            A ``(samples, sample_rate)`` tuple where *samples* is a
            float32 NumPy array normalised to ``[-1, 1]`` and
            *sample_rate* equals ``AudioLoader.TARGET_SR``.

        Raises:
            AudioLoadError: On any decoding or I/O failure.
        """
        try:
            import librosa  # lazy — not needed at import time
            wav_path = self._ensure_wav()
            audio, sr = librosa.load(wav_path, sr=self.TARGET_SR, mono=True)
            return audio, sr
        except AudioLoadError:
            raise
        except Exception as exc:
            raise AudioLoadError(f"Failed to load '{self._path.name}': {exc}") from exc
        finally:
            self._cleanup_temp()

    @property
    def file_info(self) -> str:
        """Human-readable summary: filename, size, and format."""
        size_kb = self._path.stat().st_size // 1024
        fmt = self._path.suffix.upper().lstrip(".")
        return f"{self._path.name}  ({size_kb:,} KB · {fmt})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_wav(self) -> str:
        """Return a WAV file path, converting if the source is not WAV."""
        if self._path.suffix.lower() == ".wav":
            return str(self._path)

        try:
            from pydub import AudioSegment  # type: ignore[import]
        except ImportError as exc:
            raise AudioLoadError(
                "pydub is required to convert non-WAV files.  "
                "Install it with:  pip install pydub  "
                "Then ensure ffmpeg is available:  https://ffmpeg.org/download.html"
            ) from exc

        try:
            segment = AudioSegment.from_file(str(self._path))
        except Exception as exc:
            raise AudioLoadError(
                f"pydub could not decode '{self._path.name}'.  "
                "Make sure ffmpeg is installed and on your PATH.  "
                f"Details: {exc}"
            ) from exc

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        self._temp_wav = tmp.name
        segment.export(self._temp_wav, format="wav")
        return self._temp_wav

    def _cleanup_temp(self) -> None:
        """Delete the temporary WAV file if one was created."""
        if self._temp_wav and os.path.exists(self._temp_wav):
            try:
                os.unlink(self._temp_wav)
            except OSError:
                pass
            self._temp_wav = None
