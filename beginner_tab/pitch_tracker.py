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
# Monophonic-melody post-processing filter
# ===========================================================================

# Width of each time window used by ``extract_dominant_melody``.  Notes whose
# start times fall in the same 50 ms bin are treated as "simultaneous" and
# collapsed to the single loudest note.  50 ms is short enough that a fast
# played melody (~12 notes/sec) keeps every distinct note, but long enough to
# absorb the small timing jitter that polyphonic detectors introduce when a
# single struck note produces several harmonic-related candidates.
_DOMINANT_WINDOW_S: float = 0.050  # 50 milliseconds


def extract_dominant_melody(events: list[NoteEvent]) -> list[NoteEvent]:
    """Collapse a polyphonic note list into a clean monophonic melody.

    The function bins every input event into a 50 ms time window based on its
    start time, and within each window keeps **only the single loudest note**
    (the one with the highest amplitude / confidence value).  All other notes
    sounding in that window are discarded.  The surviving notes are returned
    sorted by start time.

    This is useful when a polyphonic detector (e.g. Basic-pitch) reports
    several simultaneous notes for what was actually a single struck note,
    or when the user genuinely wants a single-line melody rendition of a
    chordal recording.

    Args:
        events: List of :data:`NoteEvent` tuples, each
            ``(start_time, midi_note, amplitude)``.  In this project a
            ``NoteEvent`` does not carry an explicit ``end_time``; notes that
            *start* in the same 50 ms bin are treated as overlapping.

    Returns:
        A new list of :data:`NoteEvent` tuples — at most one per 50 ms
        window — sorted ascending by ``start_time``.  An empty input
        produces an empty output.
    """
    if not events:
        return []

    # Group every event by its 50 ms bin index.  Using floor-division on the
    # start time means two events in the same 50 ms slice land in the same
    # bucket regardless of their exact sub-millisecond offset.
    buckets: dict[int, NoteEvent] = {}
    for event in events:
        start_time, _midi, amplitude = event
        bin_index = int(start_time // _DOMINANT_WINDOW_S)

        # Keep this event only if the bucket is empty, or if this event is
        # louder than the one currently held there.  Ties keep the earlier
        # entry, which preserves chronological order on equal-amplitude notes.
        existing = buckets.get(bin_index)
        if existing is None or amplitude > existing[2]:
            buckets[bin_index] = event

    # Sort the surviving (one-per-bucket) notes back into chronological order.
    return sorted(buckets.values(), key=lambda e: e[0])


# ===========================================================================
# Aggressive note-cleaning pipeline
# ===========================================================================
# Both pitch trackers feed their raw output through this pipeline before
# returning, so they share a single, well-tested set of filters.  Helpers are
# module-private (leading underscore) and operate on an *extended* note tuple
# that carries an explicit end_time — the public NoteEvent type does not.

# Practical 6-string guitar range in standard tuning.
GUITAR_MIDI_MIN: int = 40   # E2 — open low-E string
GUITAR_MIDI_MAX: int = 88   # E6 — practical upper bound (24 frets on high-e)

# Beginner-friendly upper bound.  Notes detected above this are octave-
# snapped DOWN rather than discarded, keeping tabs in the lower fret range.
BEGINNER_MIDI_MAX: int = 76  # E5

# Maximum gap between two consecutive same-pitch notes that should be
# treated as one sustained note rather than two re-attacks.
NOTE_MERGE_GAP_S: float = 0.10  # 100 milliseconds

# Internal tuple used during cleaning: (start_sec, end_sec, midi, amplitude).
# Distinct from the public NoteEvent type which omits end_time.
_ExtNote = tuple[float, float, int, float]


def _filter_midi_range(
    notes: list[_ExtNote],
    midi_min: int = GUITAR_MIDI_MIN,
    midi_max: int = GUITAR_MIDI_MAX,
) -> list[_ExtNote]:
    """Drop notes whose MIDI value lies outside the playable guitar range.

    Standard 6-string guitar in standard tuning spans MIDI 40 (E2, open
    low-E) through MIDI 88 (E6, ~24th fret on the high-e string).  Anything
    outside that window is almost certainly a sub-bass rumble, an
    instrument bleed (vocal, cymbal, bass), or a neural-net hallucination.

    Args:
        notes: Extended notes ``(start, end, midi, amp)``.
        midi_min: Lowest MIDI value to keep, inclusive.  Default ``40``.
        midi_max: Highest MIDI value to keep, inclusive.  Default ``88``.

    Returns:
        A new list with out-of-range notes removed.
    """
    return [n for n in notes if midi_min <= n[2] <= midi_max]


def _filter_amplitude(
    notes: list[_ExtNote], threshold: float
) -> list[_ExtNote]:
    """Drop notes whose amplitude is below ``threshold``.

    For Basic-pitch the third tuple element is the model's per-note
    amplitude estimate; for pYIN it is the voiced-frame probability.  Both
    behave like a confidence score in ``[0, 1]``, so the same threshold
    semantics apply to both backends.

    Args:
        notes: Extended notes.
        threshold: Minimum amplitude / confidence to keep, ``[0, 1]``.

    Returns:
        A new list with quiet / low-confidence notes removed.
    """
    return [n for n in notes if n[3] >= threshold]


def _filter_min_duration(
    notes: list[_ExtNote], min_duration_s: float
) -> list[_ExtNote]:
    """Drop notes shorter than ``min_duration_s`` seconds.

    Real guitar notes — even staccato 16th-notes at fast tempos — last at
    least ~60 ms.  Shorter detections are almost always pick-attack
    transients, fret-buzz blips, or polyphonic-detector artifacts.

    Args:
        notes: Extended notes.
        min_duration_s: Minimum duration in seconds.  Notes whose
            ``end - start`` is below this value are discarded.

    Returns:
        A new list with too-short notes removed.
    """
    return [n for n in notes if (n[1] - n[0]) >= min_duration_s]


def _snap_octave_down(
    notes: list[_ExtNote], max_midi: int = BEGINNER_MIDI_MAX
) -> list[_ExtNote]:
    """Transpose notes above ``max_midi`` down by full octaves until in range.

    Beginners struggle with notes above the 12th fret, so anything detected
    above MIDI 76 (E5) is dropped one octave at a time until it sits at or
    below ``max_midi``.  The ``start``, ``end``, and ``amplitude`` fields
    are preserved — only the pitch changes.

    Args:
        notes: Extended notes.
        max_midi: Highest MIDI value left untouched.  Default ``76`` (E5).

    Returns:
        A new list where every note's MIDI value is ``<= max_midi``.
    """
    out: list[_ExtNote] = []
    for start, end, midi, amp in notes:
        # Subtract octaves (12 semitones) until the note sits in range.
        while midi > max_midi:
            midi -= 12
        out.append((start, end, midi, amp))
    return out


def _merge_consecutive_same_pitch(
    notes: list[_ExtNote], max_gap_s: float = NOTE_MERGE_GAP_S
) -> list[_ExtNote]:
    """Merge adjacent same-pitch notes whose gap is shorter than ``max_gap_s``.

    Frame-based detection or brief amplitude dips can split one sustained
    pitch into several consecutive detections at the same MIDI value.
    Merging them recovers the true note duration and prevents the renderer
    from emitting artificial repeated re-strikes.

    The merged note inherits:
        * the **start_time** of the earlier note,
        * the **end_time**   of the later note,
        * the **louder**     of the two amplitudes (a conservative choice
          that keeps the merged note from being dropped by a downstream
          amplitude filter).

    Args:
        notes: Extended notes (any order).
        max_gap_s: Maximum gap in seconds between two same-pitch notes
            that qualify for merging.  ``0.10`` means a gap of strictly
            less than 100 ms triggers a merge.

    Returns:
        A new list, sorted by ``start_time``, with eligible neighbours
        combined into longer notes.
    """
    if not notes:
        return []

    notes_sorted = sorted(notes, key=lambda n: n[0])
    merged: list[_ExtNote] = [notes_sorted[0]]
    for nxt in notes_sorted[1:]:
        prev = merged[-1]
        same_pitch = nxt[2] == prev[2]
        # Note: gap can be negative if Basic-pitch reports overlapping
        # detections of the same pitch — those should also merge.
        gap = nxt[0] - prev[1]
        if same_pitch and gap < max_gap_s:
            merged[-1] = (prev[0], nxt[1], prev[2], max(prev[3], nxt[3]))
        else:
            merged.append(nxt)
    return merged


def _apply_aggressive_filters(
    notes: list[_ExtNote],
    amplitude_threshold: float,
    min_duration_s: float,
) -> list[_ExtNote]:
    """Run the full aggressive note-cleaning pipeline in spec order.

    Order of operations (chosen deliberately):

        1. **MIDI range filter** — drop anything outside MIDI 40..88.
        2. **Amplitude threshold** — drop notes quieter than the user's
           ``amplitude_threshold``.
        3. **Minimum duration** — drop notes shorter than ``min_duration_s``.
        4. **Octave snapping** — pull anything above MIDI 76 down by
           octaves into beginner-friendly territory.
        5. **Note merging** — combine same-pitch neighbours separated by
           less than 100 ms into single sustained notes.

    Octave snapping happens *before* merging on purpose: two notes that
    started in different octaves but end up at the same pitch after
    snapping are correctly merged instead of left as separate strikes.

    Args:
        notes: Extended notes ``(start, end, midi, amp)``.
        amplitude_threshold: Minimum amplitude to keep, ``[0, 1]``.
        min_duration_s: Minimum note duration in seconds.

    Returns:
        Cleaned, chronologically-sorted list of extended notes.
    """
    notes = _filter_midi_range(notes)
    notes = _filter_amplitude(notes, amplitude_threshold)
    notes = _filter_min_duration(notes, min_duration_s)
    notes = _snap_octave_down(notes)
    notes = _merge_consecutive_same_pitch(notes)
    return notes


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

    # Frequency bandpass — limits pYIN analysis to the guitar range.  FMAX
    # bumped to ~E6 so the octave-snap filter actually has high-register
    # detections to pull down (D6 was below MIDI 88).
    FMIN: float = 82.0     # E2 Hz — open low-E string (MIDI 40)
    FMAX: float = 1320.0   # E6 Hz — practical upper limit (MIDI 88)

    # MIDI range filtering is now handled centrally by
    # :func:`_apply_aggressive_filters` using ``GUITAR_MIDI_MIN`` /
    # ``GUITAR_MIDI_MAX`` (40..88).

    # Default voiced-probability threshold; lower catches more notes but
    # also more noise.  0.45 is a good balance for full-mix recordings.
    DEFAULT_CONFIDENCE: float = 0.45

    # Defaults for the aggressive cleaning pipeline (overridable per call).
    DEFAULT_AMPLITUDE_THRESHOLD: float = 0.40
    DEFAULT_MIN_DURATION_S: float = 0.06  # 60 ms

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        min_confidence: float = DEFAULT_CONFIDENCE,
        use_hpss: bool = True,
        monophonic_mode: bool = True,
        amplitude_threshold: float = DEFAULT_AMPLITUDE_THRESHOLD,
        min_duration_s: float = DEFAULT_MIN_DURATION_S,
    ) -> None:
        """Initialise PitchTracker.

        Args:
            audio: Mono float32 audio samples.
            sr: Sample rate in Hz.
            min_confidence: Frame-level voiced-probability gate ``[0, 1]``.
                Applied while iterating pYIN frames, before they are even
                grouped into proto-notes.  0.45 suits mixed recordings;
                raise to 0.7+ for clean tracks.
            use_hpss: Apply Harmonic-Percussive Source Separation before
                pYIN.  Strongly recommended for full-mix recordings.
            monophonic_mode: If ``True`` (default), pass the cleaned events
                through :func:`extract_dominant_melody` to keep only the
                loudest note per 50 ms window.
            amplitude_threshold: Note-level confidence gate ``[0, 1]`` used
                inside the aggressive cleaning pipeline.  Stacks on top of
                ``min_confidence`` — a frame must clear both to survive.
                Default ``0.40``.
            min_duration_s: Minimum note duration in seconds for the
                cleaning pipeline.  Frames shorter than this *after*
                same-pitch grouping are dropped.  Default ``0.06`` (60 ms).
        """
        self.audio = audio
        self.sr = sr
        self.min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self.use_hpss = use_hpss
        self.monophonic_mode = bool(monophonic_mode)
        self.amplitude_threshold = float(np.clip(amplitude_threshold, 0.0, 1.0))
        self.min_duration_s = max(0.0, float(min_duration_s))
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

        # --- Step 1: extract surviving voiced frames ---
        # Each surviving frame becomes ``(time, midi, voiced_prob)``.  Only
        # the frame-level confidence (``min_confidence``) is checked here;
        # the MIDI range and amplitude filters happen in the shared cleaning
        # pipeline below.
        frames: list[tuple[float, int, float]] = []
        for t, freq, voiced, prob in zip(times, f0, voiced_flag, voiced_probs):
            if voiced and freq and not np.isnan(freq) and float(prob) >= self.min_confidence:
                # Convert Hz → MIDI note number (e.g. 440 Hz → 69 = A4)
                midi = int(np.clip(round(float(librosa.hz_to_midi(freq))), 0, 127))
                frames.append((float(t), midi, float(prob)))

        # --- Step 2: group consecutive same-pitch frames into proto-notes ---
        # pYIN reports one pitch estimate per analysis frame (~23 ms).  A
        # single sustained guitar note shows up as many adjacent frames at
        # the same MIDI value, so we collapse them here before the duration
        # filter runs (otherwise every individual frame would be rejected
        # for being too short).
        hop_s = float(times[1] - times[0]) if len(times) >= 2 else 0.023
        proto_notes: list[_ExtNote] = []
        current: Optional[_ExtNote] = None
        for t, midi, prob in frames:
            if current is None:
                current = (t, t + hop_s, midi, prob)
                continue
            # Extend the running proto-note if this frame is the same pitch
            # AND directly adjacent (allow 1.5x hop to absorb pYIN jitter).
            if current[2] == midi and (t - current[1]) <= hop_s * 1.5:
                current = (current[0], t + hop_s, midi, max(current[3], prob))
            else:
                proto_notes.append(current)
                current = (t, t + hop_s, midi, prob)
        if current is not None:
            proto_notes.append(current)

        # --- Step 3: run the shared aggressive cleaning pipeline ---
        cleaned = _apply_aggressive_filters(
            proto_notes,
            amplitude_threshold=self.amplitude_threshold,
            min_duration_s=self.min_duration_s,
        )

        # --- Step 4: flatten back to the public NoteEvent shape ---
        events: list[NoteEvent] = [(s, m, a) for (s, _e, m, a) in cleaned]

        # Optional monophonic-melody post-filter: collapse any remaining
        # near-simultaneous detections (within 50 ms) to the single loudest
        # note.  Disabled when the caller wants the raw events.
        if self.monophonic_mode:
            events = extract_dominant_melody(events)
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

    # Frequency bandpass — passed to basic-pitch's predict() to limit
    # analysis.  FMAX widened to ~E6 so the octave-snap stage in the shared
    # cleaning pipeline actually has high-register notes to pull down.
    FMIN: float = 82.0     # E2 Hz — open low-E string (MIDI 40)
    FMAX: float = 1320.0   # E6 Hz — practical upper limit (MIDI 88)

    # Defaults for the aggressive cleaning pipeline (overridable per call).
    # MIDI range filtering is centralised in :func:`_apply_aggressive_filters`
    # using ``GUITAR_MIDI_MIN`` / ``GUITAR_MIDI_MAX`` (40..88).
    DEFAULT_AMPLITUDE_THRESHOLD: float = 0.40
    DEFAULT_MIN_DURATION_S: float = 0.06  # 60 milliseconds

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length_ms: float = 80.0,
        monophonic_mode: bool = True,
        amplitude_threshold: float = DEFAULT_AMPLITUDE_THRESHOLD,
        min_duration_s: float = DEFAULT_MIN_DURATION_S,
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
            minimum_note_length_ms: Shortest note Basic-pitch itself will
                emit (milliseconds).  Acts inside the model; the cleaning
                pipeline below applies its own duration filter on top.
            monophonic_mode: If ``True`` (default), pass the cleaned events
                through :func:`extract_dominant_melody` so each 50 ms slice
                retains only the loudest note.  Set ``False`` to keep the
                full polyphonic note set (e.g. for chord rendering).
            amplitude_threshold: Minimum note amplitude kept by the shared
                aggressive cleaning pipeline, ``[0, 1]``.  Default ``0.40``.
                Stricter than Basic-pitch's internal thresholds — drops
                instrument bleed and detection artifacts.
            min_duration_s: Minimum note duration kept by the cleaning
                pipeline, in seconds.  Default ``0.06`` (60 ms).
        """
        self.audio = audio
        self.sr = sr
        self.onset_threshold = float(np.clip(onset_threshold, 0.0, 1.0))
        self.frame_threshold = float(np.clip(frame_threshold, 0.0, 1.0))
        self.minimum_note_length_ms = float(minimum_note_length_ms)
        self.monophonic_mode = bool(monophonic_mode)
        self.amplitude_threshold = float(np.clip(amplitude_threshold, 0.0, 1.0))
        self.min_duration_s = max(0.0, float(min_duration_s))
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
            from basic_pitch.inference import predict
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
            with os.fdopen(tmp_fd, "wb"):
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

        # --- Build the extended-note list for the cleaning pipeline ---
        # Each raw note is a tuple:
        #   (start_sec, end_sec, midi_int, amplitude, pitch_bends)
        # We keep only the four fields the pipeline cares about.  MIDI is
        # clipped to a valid 0..127 byte just in case the model emits a
        # nonsense value; the range filter then narrows it to 40..88.
        proto_notes: list[_ExtNote] = [
            (float(n[0]), float(n[1]), int(np.clip(int(n[2]), 0, 127)), float(n[3]))
            for n in raw_notes
        ]

        # --- Run the shared aggressive cleaning pipeline ---
        # Range-filter → amplitude → min-duration → octave-snap → merge.
        cleaned = _apply_aggressive_filters(
            proto_notes,
            amplitude_threshold=self.amplitude_threshold,
            min_duration_s=self.min_duration_s,
        )

        # --- Flatten back to the public NoteEvent shape ---
        events: list[NoteEvent] = [(s, m, a) for (s, _e, m, a) in cleaned]

        # Optional monophonic-melody post-filter.  When enabled, any 50 ms
        # window keeps only the single loudest note — turning a polyphonic
        # detection into a clean single-line melody for the standard
        # (non-chord) tab pipeline.
        if self.monophonic_mode:
            events = extract_dominant_melody(events)
        return events

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
