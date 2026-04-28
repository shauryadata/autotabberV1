# ==========================================================================
# File:        source_separator.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-27
# Description: Optional preprocessing stage that runs Spotify Demucs (the
#              4-stem htdemucs model by default) on an input file and
#              returns a single requested stem — vocals, drums, bass, or
#              "other" — as a mono float32 numpy array.  Useful before
#              pitch tracking when the user uploads a full mix and wants
#              the guitar (typically the "other" stem) isolated from
#              vocals / drums / bass before AutoTabber transcribes it.
#
# Why a separate module:
#     Demucs depends on PyTorch and downloads ~80 MB of model weights on
#     first use.  We keep all of that strictly opt-in: the heavy imports
#     and model load only happen the first time :meth:`separate` is
#     called, and the module is excluded from the Streamlit Cloud build
#     (`requirements-local.txt` only).
#
# API note:
#     The original spec referenced ``demucs.separate.Separator``, but
#     that class does not exist in any released demucs version (it's a
#     post-4.0.1 feature still under development).  We implement the
#     same public surface using the canonical lower-level API:
#     ``demucs.pretrained.get_model`` + ``demucs.apply.apply_model``.
#     The module-level ``import demucs`` makes the missing-dependency
#     path raise an ImportError at module load, which is what
#     :mod:`beginner_tab.__init__` relies on for conditional export.
# ==========================================================================

"""Optional Demucs source-separation preprocessing stage.

Public API:
    * :class:`SourceSeparator`     — main class.
    * :class:`SourceSeparationError` — single exception type for any
      failure (missing model, malformed audio, torch / CUDA error, …).

Typical usage::

    from beginner_tab import SourceSeparator
    sep = SourceSeparator()                              # cheap — model NOT yet loaded
    audio, sr = sep.separate("song.mp3", target_stem="other")
    # Now feed (audio, sr) into PitchTracker / BasicPitchTracker.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

# Probe demucs at import time so :mod:`beginner_tab.__init__` can detect
# its absence and skip the conditional export per the spec.  This is the
# *only* eager demucs import — every other demucs / torch import is
# deferred to :meth:`SourceSeparator._load_model` so the module can be
# referenced (e.g. for type hints, ``is_available`` checks) without
# paying the heavy startup cost.
import demucs  # noqa: F401  — presence check; deeper imports are lazy

# Module-level logger for progress messages.  Demucs separation takes
# 30-90 s on CPU even for short clips, so users need feedback.  Callers
# can wire this into Streamlit (``st.info``) by adding a logging handler.
log = logging.getLogger(__name__)


class SourceSeparationError(Exception):
    """Single error type raised by :class:`SourceSeparator`.

    Wraps any underlying failure — missing model file, malformed audio,
    torch / CUDA error, unknown stem name, etc. — into a clear,
    user-actionable message.  Always raised with ``raise … from exc`` so
    the original traceback is preserved for debugging.
    """


class SourceSeparator:
    """Run Demucs source separation and return a single isolated stem.

    The class is deliberately lightweight to construct: ``__init__`` does
    *not* load the Demucs model.  The ~80 MB model weights are loaded the
    first time :meth:`separate` is called and cached on the instance for
    subsequent calls.

    Example::

        sep = SourceSeparator(model_name="htdemucs")
        # ... cheap so far, no model in memory ...
        vocal_audio, sr = sep.separate("clip.mp3", target_stem="vocals")
        # The model is now loaded and resident; the next call is fast.
        guitar_audio, _ = sep.separate("clip.mp3", target_stem="other")

    Attributes:
        model_name: Name of the Demucs pretrained model to use.
        AVAILABLE_STEMS: Class constant — the four stems htdemucs (and
            every other 4-stem Demucs model) emits.
    """

    # Canonical 4-stem set produced by every Demucs default model
    # (htdemucs, mdx, mdx_extra, …).  Listed in the order the spec asked
    # for, not the order Demucs internally uses.
    AVAILABLE_STEMS: tuple[str, ...] = ("vocals", "drums", "bass", "other")

    def __init__(self, model_name: str = "htdemucs") -> None:
        """Construct a SourceSeparator (does **not** load the model).

        Args:
            model_name: Demucs pretrained model name.  Defaults to
                ``"htdemucs"`` — the recommended 4-stem hybrid
                transformer model.  Other valid options include
                ``"htdemucs_ft"`` (fine-tuned) and ``"mdx_extra"``.
        """
        self.model_name = model_name

        # Lazy-loaded state.  All four are populated by ``_load_model``.
        self._model = None              # the demucs model object
        self._sample_rate: Optional[int] = None
        self._channels: Optional[int] = None
        self._sources: Optional[list[str]] = None  # actual stem names from model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_available_stems(cls) -> list[str]:
        """Return the list of stem names this separator can extract.

        These are the canonical four stems produced by every standard
        Demucs 4-stem model.  Returned as a fresh list (not the class
        constant tuple) so callers can mutate it freely.

        Returns:
            ``["vocals", "drums", "bass", "other"]``.
        """
        return list(cls.AVAILABLE_STEMS)

    @classmethod
    def is_available(cls) -> bool:
        """Return True if Demucs and its dependencies can be imported.

        Used by the UI layer to hide / disable the source-separation
        feature on systems where demucs is not installed (e.g. the
        Streamlit Cloud build).  This check is intentionally cheap — it
        verifies imports succeed but does *not* download or load the
        model.  A True return therefore guarantees that ``separate``
        will not fail at the *import* step; it can still fail later if
        the model download is blocked or the audio file is malformed.

        Returns:
            bool: True if ``demucs.pretrained``, ``demucs.apply`` and
            ``torch`` all import cleanly; False on any ImportError.
        """
        try:
            import demucs.pretrained  # noqa: F401
            import demucs.apply       # noqa: F401
            import torch              # noqa: F401
            return True
        except Exception:  # noqa: BLE001 — broad on purpose; any error == unavailable
            return False

    def separate(
        self, audio_path: str, target_stem: str
    ) -> tuple[np.ndarray, int]:
        """Run Demucs and return a single isolated stem as mono audio.

        Args:
            audio_path: Filesystem path to the input audio file.  Any
                format Demucs / FFmpeg can read (MP3, WAV, M4A, FLAC,
                OGG, …) is accepted.
            target_stem: Which stem to extract.  Must be one of
                :attr:`AVAILABLE_STEMS` (``"vocals"``, ``"drums"``,
                ``"bass"``, ``"other"``).  For guitar transcription the
                most useful stem is usually ``"other"`` (which contains
                guitar, keys, and any non-vocal melodic content).

        Returns:
            ``(audio, sample_rate)`` where:
                * ``audio`` is a 1-D ``float32`` numpy array — the
                  requested stem downmixed to mono.  Output of Demucs
                  is stereo; we average left and right channels so the
                  result drops straight into AutoTabber's existing
                  mono-only pitch trackers.
                * ``sample_rate`` is the model's native rate
                  (44100 Hz for htdemucs).  No resampling is performed.

        Raises:
            SourceSeparationError: On any failure — unknown stem name,
                missing audio file, model download blocked, torch / CUDA
                error during inference, etc.  The original exception is
                preserved as ``__cause__``.
        """
        # --- Validate the requested stem before doing anything expensive ---
        if target_stem not in self.AVAILABLE_STEMS:
            raise SourceSeparationError(
                f"Unknown target_stem {target_stem!r}; "
                f"must be one of {list(self.AVAILABLE_STEMS)}."
            )

        # --- Lazy model load (cached after first call) ---
        try:
            self._load_model()
        except SourceSeparationError:
            raise
        except Exception as exc:
            raise SourceSeparationError(
                f"Failed to load Demucs model {self.model_name!r}: {exc}"
            ) from exc

        # The actual model may produce a different stem set than the
        # canonical 4 (e.g. htdemucs_6s adds "guitar" and "piano").  Be
        # explicit about what's missing rather than failing obscurely.
        assert self._sources is not None  # populated by _load_model
        if target_stem not in self._sources:
            raise SourceSeparationError(
                f"Stem {target_stem!r} is not produced by model "
                f"{self.model_name!r}; this model emits {self._sources}."
            )

        # --- Inference ---
        try:
            return self._run_inference(audio_path, target_stem)
        except SourceSeparationError:
            raise
        except Exception as exc:
            # Catch torch / CUDA / OOM / decoding errors and surface a
            # clean message; keep the original traceback as __cause__.
            raise SourceSeparationError(
                f"Demucs separation failed for {audio_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Demucs model and cache it on the instance.

        Idempotent — subsequent calls are no-ops once the model is
        cached.  All heavy imports (``demucs.pretrained``,
        ``demucs.apply``, ``torch``) live inside this method so simply
        importing the module stays cheap.

        Raises:
            SourceSeparationError: If model loading fails for any reason.
        """
        if self._model is not None:
            return

        log.info("Loading Demucs model %r — this may download ~80 MB on first use.",
                 self.model_name)
        try:
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise SourceSeparationError(
                "demucs.pretrained could not be imported — install with "
                "`pip install demucs`."
            ) from exc

        model = get_model(self.model_name)
        # demucs returns a BagOfModels for the default htdemucs name,
        # which exposes the same .samplerate / .audio_channels / .sources
        # attributes as a plain Demucs model.
        self._model = model
        self._sample_rate = int(model.samplerate)
        self._channels = int(model.audio_channels)
        self._sources = list(model.sources)
        log.info(
            "Loaded Demucs model %r: sources=%s, sr=%d Hz, channels=%d",
            self.model_name, self._sources, self._sample_rate, self._channels,
        )

    def _load_audio_as_tensor(self, audio_path: str):
        """Load *audio_path* and return a torch tensor of shape (C, N).

        Bypasses ``demucs.audio.AudioFile`` to avoid a numpy/torch ABI
        bug (``torch.from_numpy`` raising "expected np.ndarray (got
        numpy.ndarray)" on certain conda / pip pairings).  The returned
        tensor sits at the model's native sample rate
        (``self._sample_rate``) and channel count (``self._channels``).

        Args:
            audio_path: Path to any format ``soundfile`` can read (WAV,
                FLAC, OGG natively; MP3 / M4A via libsndfile ≥ 1.1.0).

        Returns:
            ``torch.Tensor`` of shape ``(channels, samples)``,
            ``dtype=float32``.

        Raises:
            FileNotFoundError: If ``audio_path`` does not exist.
            Any exception raised by soundfile / librosa during decode —
            wrapped by the calling :meth:`separate` into a
            :class:`SourceSeparationError`.
        """
        import soundfile as sf
        import torch
        import librosa

        assert self._sample_rate is not None
        assert self._channels is not None

        # soundfile returns (samples, channels) when always_2d=True.
        data, file_sr = sf.read(audio_path, dtype="float32", always_2d=True)
        # → (channels, samples) for the rest of the pipeline.
        data = data.T

        # Channel adaptation: htdemucs wants exactly self._channels
        # (typically 2).  Mono input is duplicated; >2-channel input is
        # truncated to the first ``self._channels``.
        if data.shape[0] == 1 and self._channels > 1:
            data = np.repeat(data, self._channels, axis=0)
        elif data.shape[0] > self._channels:
            data = data[: self._channels]

        # Resample if needed.  librosa.resample expects shape (..., samples)
        # and broadcasts across leading dims, so a (channels, samples)
        # input round-trips correctly.
        if file_sr != self._sample_rate:
            data = librosa.resample(
                data, orig_sr=file_sr, target_sr=self._sample_rate, axis=-1
            )

        # ``torch.as_tensor`` shares memory where possible and — unlike
        # ``torch.from_numpy`` — does not trip the numpy ABI bug.
        return torch.as_tensor(data, dtype=torch.float32)

    def _run_inference(
        self, audio_path: str, target_stem: str
    ) -> tuple[np.ndarray, int]:
        """Internal: load audio, run apply_model, return mono stem.

        Split out from :meth:`separate` so the wrapper can apply blanket
        exception handling without losing readability.  This method is
        the only place that touches torch tensors.

        Args:
            audio_path: Path to the input audio.
            target_stem: Validated stem name (presence in
                ``self._sources`` already checked by caller).

        Returns:
            ``(mono_float32_audio, sample_rate)``.

        Raises:
            Any exception — the caller wraps these in
            :class:`SourceSeparationError`.
        """
        import torch
        from demucs.apply import apply_model

        assert self._model is not None
        assert self._sample_rate is not None
        assert self._channels is not None
        assert self._sources is not None

        # --- Load + resample + reshape to (channels, samples) ---
        # We deliberately *do not* use ``demucs.audio.AudioFile`` here:
        # that helper calls ``torch.from_numpy`` internally, which raises
        # ``TypeError: expected np.ndarray (got numpy.ndarray)`` on any
        # combination where torch was compiled against a different numpy
        # ABI than the one in the runtime environment (a common conda /
        # pip pairing problem).  Loading via soundfile + librosa.resample
        # and going through ``torch.as_tensor`` (which uses a different
        # code path) sidesteps the bug entirely and works on every
        # reasonable numpy/torch combination.
        log.info("Loading audio: %s", audio_path)
        wav = self._load_audio_as_tensor(audio_path)
        # ``wav`` is a torch tensor of shape (channels, samples) at the
        # model's native sample rate and channel count.
        n_samples = wav.shape[-1]
        log.info(
            "Running Demucs separation on %.1f s of audio — this is the slow step (30-90 s on CPU).",
            n_samples / self._sample_rate,
        )

        # --- Per-channel normalisation (matches demucs CLI behaviour) ---
        # Demucs is sensitive to input levels; we normalise to roughly
        # zero mean and unit variance per the recipe in the demucs CLI,
        # then reverse the scaling on the output so the returned audio
        # sits at the same loudness as the input.
        ref = wav.mean(0)                    # mono reference
        wav_norm = (wav - ref.mean()) / max(ref.std().item(), 1e-8)

        # apply_model expects shape (batch, channels, samples).
        mix = wav_norm.unsqueeze(0)
        with torch.no_grad():
            sources = apply_model(
                self._model,
                mix,
                shifts=1,         # number of random equivariant shifts
                split=True,       # process in chunks (memory-efficient)
                overlap=0.25,     # 25% overlap between chunks
                progress=False,   # disable tqdm — we use logging instead
                num_workers=0,    # avoid extra processes inside Streamlit
            )
        # ``sources`` shape: (batch=1, n_stems, channels, samples).
        sources = sources * max(ref.std().item(), 1e-8) + ref.mean()

        stem_idx = self._sources.index(target_stem)
        stem_tensor = sources[0, stem_idx]   # (channels, samples)

        # --- Downmix stereo → mono for the existing pitch trackers ---
        # AutoTabber's PitchTracker / BasicPitchTracker both expect a
        # 1-D float32 array.  Averaging is the standard downmix.
        stem_np = stem_tensor.detach().cpu().numpy()
        if stem_np.ndim == 2 and stem_np.shape[0] > 1:
            mono = stem_np.mean(axis=0)
        else:
            mono = stem_np[0] if stem_np.ndim == 2 else stem_np

        log.info(
            "Separation complete: %r stem extracted (%d samples @ %d Hz)",
            target_stem, mono.shape[0], self._sample_rate,
        )
        return mono.astype(np.float32, copy=False), self._sample_rate
