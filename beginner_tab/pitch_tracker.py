# ==========================================================================
# File:        pitch_tracker.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Provides two pitch-detection backends that convert raw audio
#              into timestamped MIDI note events.  PitchTracker uses the
#              classical pYIN algorithm (monophonic, one note at a time).
#              BasicPitchTracker uses Spotify's convolutional neural network
#              (polyphonic, detects chords).  Both return the same NoteEvent
#              tuple type so the rest of the pipeline is backend-agnostic.
# ==========================================================================

"""Pitch detection backends — pYIN (monophonic) and Basic-pitch (polyphonic).

Two tracker classes share the same ``NoteEvent`` output type so the rest of
the pipeline is backend-agnostic.

PitchTracker (pYIN)
    Classic probabilistic YIN algorithm via librosa.  Single note at a time.
    Fast, no model download.  Works well on isolated melody recordings.
    Optional HPSS pre-processing removes drum hits before detection.

BasicPitchTracker (Spotify Basic-pitch)
    Convolutional neural network trained on real music.  Returns *polyphonic*
    note events — multiple simultaneous notes (chords) are supported.
    Requires tensorflow (installed with basic-pitch).  First call loads the
    model (~200 MB) from disk; subsequent calls are fast.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# scipy compatibility shim
# ---------------------------------------------------------------------------
# scipy >=1.14 moved scipy.signal.gaussian → scipy.signal.windows.gaussian
# and removed the old location.  basic-pitch's internals still reference the
# old path, which causes an AttributeError at inference time.  This shim
# patches the missing attribute so basic-pitch works on any scipy version.
try:
    import scipy.signal
    if not hasattr(scipy.signal, "gaussian"):
        from scipy.signal.windows import gaussian as _gaussian
        scipy.signal.gaussian = _gaussian  # type: ignore[attr-defined]
except ImportError:
    pass  # scipy not installed — will fail later with a clearer message

# ---------------------------------------------------------------------------
# Shared type alias used by every downstream pipeline stage.
# (time_in_seconds, MIDI_note_0-127, confidence_or_amplitude_0-1)
# Multiple events sharing the same timestamp = simultaneous chord notes.
# ---------------------------------------------------------------------------
NoteEvent = tuple[float, int, float]


class PitchDetectionError(Exception):
    """Raised when any pitch-detection backend fails."""


# ===========================================================================
# pYIN tracker (monophonic)
# ===========================================================================

class PitchTracker:
    """Monophonic pitch tracker using librosa's pYIN algorithm.

    Optional HPSS pre-processing separates harmonic content (melody, chords)
    from percussive content (drums) before running pYIN, significantly
    improving note detection on mixed recordings.

    Example::

        tracker = PitchTracker(audio, sr, use_hpss=True)
        notes  = tracker.track()
        tempo  = tracker.estimate_tempo()
    """

    # Frequency bandpass — limits pYIN analysis to the guitar range only
    FMIN: float = 82.0     # E2 Hz — lowest open string on a standard guitar
    FMAX: float = 1175.0   # D6 Hz — practical upper limit for guitar fretboard

    # MIDI range filter — any detected note outside these bounds is discarded
    MIDI_MIN: int = 28     # MIDI 28 ≈ E2 (lowest guitar note)
    MIDI_MAX: int = 74     # MIDI 74 ≈ D6 (highest practical guitar note)

    # Default voiced-probability threshold; lower catches more notes but
    # also more noise.  0.45 is a good balance for full-mix recordings.
    DEFAULT_CONFIDENCE: float = 0.45

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        min_confidence: float = DEFAULT_CONFIDENCE,
        use_hpss: bool = True,
    ) -> None:
        """Initialise PitchTracker.

        Args:
            audio: Mono float32 audio samples.
            sr: Sample rate in Hz.
            min_confidence: Voiced-probability threshold ``[0, 1]``.
                0.45 suits mixed recordings; raise to 0.7+ for clean tracks.
            use_hpss: Apply Harmonic-Percussive Source Separation before
                pYIN.  Strongly recommended for full-mix recordings.
        """
        self.audio = audio
        self.sr = sr
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.use_hpss = use_hpss
        self._tempo: Optional[float] = None

    def track(self) -> list[NoteEvent]:
        """Run pYIN and return voiced note events sorted by time.

        Returns:
            List of ``(time_sec, midi_note, confidence)`` tuples.

        Raises:
            PitchDetectionError: On any librosa failure.
        """
        try:
            import librosa  # lazy import — only loaded when track() is called

            src = self.audio
            if self.use_hpss:
                # HPSS: split audio into harmonic (melody/chords) and percussive
                # (drums/transients) components.  Feed only the harmonic part
                # into pYIN so drum hits don't create false pitch detections.
                harmonic, _ = librosa.effects.hpss(src)
                src = harmonic

            # pYIN returns three parallel arrays, one value per analysis frame
            # (~23 ms each at 22 050 Hz):
            #   f0           — estimated fundamental frequency in Hz (NaN if unvoiced)
            #   voiced_flag  — boolean, True if the frame is voiced
            #   voiced_probs — float [0,1], probability that the frame is voiced
            f0, voiced_flag, voiced_probs = librosa.pyin(
                src, fmin=self.FMIN, fmax=self.FMAX, sr=self.sr
            )
        except ImportError as exc:
            raise PitchDetectionError("librosa not installed: pip install librosa") from exc
        except Exception as exc:
            raise PitchDetectionError(f"pYIN failed: {exc}") from exc

        # Convert frame indices to time in seconds
        times = librosa.times_like(f0, sr=self.sr)

        events: list[NoteEvent] = []
        for t, freq, voiced, prob in zip(times, f0, voiced_flag, voiced_probs):
            # Keep only frames that are voiced, have a valid frequency,
            # and meet the user's confidence threshold
            if voiced and freq and not np.isnan(freq) and float(prob) >= self.min_confidence:
                # Convert Hz → MIDI note number (e.g. 440 Hz → 69 = A4)
                midi = int(np.clip(round(float(librosa.hz_to_midi(freq))), 0, 127))
                # Discard notes outside the guitar's playable MIDI range
                if self.MIDI_MIN <= midi <= self.MIDI_MAX:
                    events.append((float(t), midi, float(prob)))
        return events

    def estimate_tempo(self) -> float:
        """Estimate tempo in BPM (cached).  Falls back to 120.0 on error.

        Uses librosa's beat tracker which analyses the onset strength
        envelope of the audio.  The result is cached so repeated calls
        do not recompute.

        Returns:
            float: Estimated beats per minute, or 120.0 as a safe default.
        """
        if self._tempo is not None:
            return self._tempo
        try:
            import librosa
            # beat_track returns (tempo, beat_frames); we only need tempo [0]
            raw = librosa.beat.beat_track(y=self.audio, sr=self.sr)[0]
            # librosa may return a scalar or a 1-element array depending on version
            bpm = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
            self._tempo = bpm if bpm > 0 else 120.0
        except Exception:
            self._tempo = 120.0  # safe default if beat tracking fails
        return self._tempo


# ===========================================================================
# Basic-pitch tracker (polyphonic, neural network)
# ===========================================================================

class BasicPitchTracker:
    """Polyphonic pitch tracker powered by Spotify's Basic-pitch model.

    Basic-pitch uses a lightweight convolutional neural network (CNN) trained
    on diverse real-music recordings.  It detects *multiple simultaneous
    notes*, making it suitable for guitar chords and full-mix audio.

    The returned ``NoteEvent`` list may contain several events sharing the
    same ``time_sec`` — these represent simultaneously sounding notes
    (a chord).  Pass the result to ``TabSimplifier.quantize_chords()`` to
    group them correctly.

    The TensorFlow model is loaded on the first call to :meth:`track` and
    cached for the lifetime of the process.  This adds ~5-10 s overhead on
    the first run; subsequent calls are fast.

    Example::

        tracker = BasicPitchTracker(audio, sr, onset_threshold=0.5)
        notes  = tracker.track()         # polyphonic NoteEvents
        tempo  = tracker.estimate_tempo()
    """

    # Frequency bandpass — passed to basic-pitch's predict() to limit analysis
    FMIN: float = 82.0     # E2 Hz — lowest open string on a standard guitar
    FMAX: float = 1175.0   # D6 Hz — practical upper limit for guitar fretboard

    # MIDI range filter — any note outside these bounds is discarded post-inference
    MIDI_MIN: int = 28     # MIDI 28 ≈ E2 (lowest guitar note)
    MIDI_MAX: int = 74     # MIDI 74 ≈ D6 (highest practical guitar note)

    # Amplitude gate — notes below this level are likely background noise or
    # bleed from other instruments (e.g. bass, vocals), not intentional guitar
    MIN_AMPLITUDE: float = 0.35

    # Duration gate — notes shorter than 60 ms are almost certainly artifacts
    # from transient noise, not real played notes
    MIN_DURATION_S: float = 0.06  # 60 milliseconds

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 80.0,
    ) -> None:
        """Initialise BasicPitchTracker.

        Args:
            audio: Mono float32 audio samples.
            sr: Sample rate in Hz.
            onset_threshold: Confidence required to start a new note
                ``[0, 1]``.  Higher = fewer but more confident notes.
                Default 0.5 is a good starting point for mixed recordings.
            frame_threshold: Confidence to sustain a note across frames
                ``[0, 1]``.  Usually kept below ``onset_threshold``.
            minimum_note_length_ms: Shortest note kept (milliseconds).
                80 ms ≈ a 32nd note at 93 BPM — filters sub-note blips.
        """
        self.audio = audio
        self.sr = sr
        self.onset_threshold = float(np.clip(onset_threshold, 0.0, 1.0))
        self.frame_threshold = float(np.clip(frame_threshold, 0.0, 1.0))
        self.minimum_note_length_ms = maximum = float(minimum_note_length_ms)
        self._tempo: Optional[float] = None

    def track(self) -> list[NoteEvent]:
        """Run Basic-pitch inference and return polyphonic note events.

        The audio is written to a temporary WAV file (required by the
        Basic-pitch API), inference is run, and the file is cleaned up.

        Each ``NoteEvent`` is ``(start_time_sec, midi_note, amplitude)``.
        Multiple events at the same ``start_time_sec`` represent a chord.

        Returns:
            List of :data:`NoteEvent` tuples sorted by time.

        Raises:
            PitchDetectionError: If basic-pitch or tensorflow are missing,
                or if inference fails.
        """
        # Lazy imports — these heavy libraries are only loaded when inference
        # is actually requested, keeping test imports fast.
        #
        # The entire import + model-resolution block is wrapped in a single
        # try/except so that ANY failure (missing package, missing model file,
        # incompatible TF version, etc.) produces one clear error message
        # instead of an obscure traceback.
        try:
            import soundfile as sf
            from basic_pitch.inference import predict, Model
            from basic_pitch import ICASSP_2022_MODEL_PATH

            # --- Model backend resolution ---
            # basic-pitch 0.3's default ICASSP_2022_MODEL_PATH points to the
            # TensorFlow SavedModel directory ("nmp"), which can fail on
            # TF >=2.16 due to a '_UserObject' serialisation change.  If the
            # lighter ONNX Runtime is installed, we swap to the "nmp.onnx"
            # file instead — same weights, better cross-version compatibility.
            import pathlib
            model_path = pathlib.Path(ICASSP_2022_MODEL_PATH)  # default: .../nmp
            onnx_path = model_path.with_suffix(".onnx")        # .../nmp.onnx
            try:
                import onnxruntime  # noqa: F401
                if onnx_path.exists():
                    model_path = onnx_path
            except ImportError:
                pass  # fall back to default TF / TFLite backend

            # Verify the resolved model path actually exists on disk
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")

        except Exception:
            raise PitchDetectionError(
                "Basic-pitch model unavailable in this environment. "
                "Please use pYIN detector instead."
            )

        # --- Write temp WAV for inference ---
        # basic-pitch's predict() requires a file path, not an in-memory array.
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pass  # close the fd so soundfile can open the path for writing
            sf.write(tmp_path, self.audio, self.sr)

            # predict() returns (model_output_dict, PrettyMIDI, raw_note_list)
            # We only need raw_note_list (index [2]).
            _, _, raw_notes = predict(
                tmp_path,
                model_path,
                onset_threshold=self.onset_threshold,
                frame_threshold=self.frame_threshold,
                minimum_note_length=self.minimum_note_length_ms,
                minimum_frequency=self.FMIN,
                maximum_frequency=self.FMAX,
            )
        except PitchDetectionError:
            raise
        except Exception as exc:
            raise PitchDetectionError(f"Basic-pitch inference failed: {exc}") from exc
        finally:
            # Always clean up the temporary file, even on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # --- Post-processing filters ---
        # Each raw note is a tuple: (start_sec, end_sec, midi_int, amplitude, pitch_bends)
        events: list[NoteEvent] = []
        for note in raw_notes:
            start_time = float(note[0])  # note onset in seconds
            end_time = float(note[1])    # note offset in seconds
            midi = int(note[2])          # MIDI note number (0-127)
            amplitude = float(note[3])   # detection amplitude / confidence

            # Filter 1: Duration — discard notes shorter than MIN_DURATION_S
            # (likely transient artifacts, not real played notes)
            if (end_time - start_time) < self.MIN_DURATION_S:
                continue

            # Filter 2: Amplitude — discard quiet notes below MIN_AMPLITUDE
            # (likely bleed from other instruments or background noise)
            if amplitude < self.MIN_AMPLITUDE:
                continue

            # Filter 3: Guitar range — discard notes outside MIDI_MIN..MIDI_MAX
            # (non-guitar frequencies that leaked through the neural network)
            midi = int(np.clip(midi, 0, 127))
            if not (self.MIDI_MIN <= midi <= self.MIDI_MAX):
                continue

            events.append((start_time, midi, amplitude))

        # Return events sorted chronologically for downstream processing
        return sorted(events, key=lambda e: e[0])

    def estimate_tempo(self) -> float:
        """Estimate tempo in BPM using librosa's beat tracker (cached).

        Falls back to 120.0 BPM on any error so the pipeline never stalls.

        Returns:
            float: Estimated beats per minute, or 120.0 as a safe default.
        """
        if self._tempo is not None:
            return self._tempo
        try:
            import librosa
            # beat_track returns (tempo, beat_frames); we only need tempo [0]
            raw = librosa.beat.beat_track(y=self.audio, sr=self.sr)[0]
            bpm = float(raw[0]) if hasattr(raw, "__len__") else float(raw)
            self._tempo = bpm if bpm > 0 else 120.0
        except Exception:
            self._tempo = 120.0  # safe default if beat tracking fails
        return self._tempo
