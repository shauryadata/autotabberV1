"""Standalone smoke-test for beginner_tab.source_separator.

Run from the repo root::

    python3 scripts/verify_source_separator.py [path/to/clip.mp3]

If no path is given the script synthesises a short stereo test signal
(a low sine + a higher sine + white noise) and writes it to a temp WAV
so the rest of the script can run without external audio.  Note that
synthetic audio is **not** what Demucs is trained on, so the perceptual
quality of the four output stems will be poor — what we are verifying
here is the *plumbing*: model load, tensor handoff, stem extraction,
mono downmix, file write.  Run with a real MP3 to get meaningful stems.

Side effects:
    Writes one WAV per stem to a fresh temp directory and prints the
    path so the operator can listen to them.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Make ``beginner_tab`` importable when this script is run from the repo
# root without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from beginner_tab import (  # noqa: E402  — sys.path edit must run first
    SOURCE_SEPARATOR_AVAILABLE,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Demucs has a minimum useful chunk size around 7-8 s; use 10 s to stay
# safely above that for the synthetic fallback signal.
_SYNTH_DURATION_S = 10.0
_SYNTH_SAMPLE_RATE = 44100  # match htdemucs native rate to avoid resampling


def _make_synthetic_clip(path: Path) -> None:
    """Write a 10-second stereo test signal to ``path``.

    The signal is intentionally simple — it lets us verify that Demucs
    runs end-to-end without requiring a real music file.  Quality of
    the resulting "vocals" / "drums" / "bass" / "other" stems on this
    input is not meaningful.

    Args:
        path: WAV destination.  Parent directory must exist.
    """
    sr = _SYNTH_SAMPLE_RATE
    t = np.linspace(0, _SYNTH_DURATION_S, int(sr * _SYNTH_DURATION_S),
                    endpoint=False)
    # 80 Hz sine ≈ low bass-ish thump
    bass = 0.30 * np.sin(2 * np.pi * 80.0 * t)
    # 440 Hz sine ≈ harmonic content
    melody = 0.30 * np.sin(2 * np.pi * 440.0 * t)
    # Soft white noise ≈ percussion
    rng = np.random.default_rng(seed=0)
    noise = 0.10 * rng.standard_normal(t.shape).astype(np.float32)
    mix = (bass + melody + noise).astype(np.float32)
    # Stereo by duplication (Demucs is trained on stereo)
    stereo = np.stack([mix, mix], axis=-1)
    sf.write(path, stereo, sr)


def main() -> int:
    """Drive the smoke-test.  Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio_path", nargs="?", default=None,
        help="Optional path to an audio file. Synthesises one if omitted.",
    )
    parser.add_argument(
        "--model", default="htdemucs",
        help="Demucs model name (default: htdemucs).",
    )
    args = parser.parse_args()

    # Surface Demucs progress messages from beginner_tab.source_separator.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- 1. Confirm the conditional export saw demucs ---
    if not SOURCE_SEPARATOR_AVAILABLE:
        print("ERROR: SOURCE_SEPARATOR_AVAILABLE is False — demucs is not "
              "installed in this environment.  Install with:\n"
              "    pip install -r requirements-local.txt", file=sys.stderr)
        return 2

    # Import lazily so the unavailable-path message above stays clean.
    from beginner_tab import SourceSeparator, SourceSeparationError

    # --- 2. is_available() and get_available_stems() smoke checks ---
    assert SourceSeparator.is_available() is True, "is_available should be True"
    stems = SourceSeparator.get_available_stems()
    assert stems == ["vocals", "drums", "bass", "other"], stems
    print(f"is_available() → True; stems = {stems}")

    # --- 3. Prepare an input clip ---
    workdir = Path(tempfile.mkdtemp(prefix="autotab_sep_"))
    if args.audio_path:
        audio_path = Path(args.audio_path).expanduser().resolve()
        if not audio_path.is_file():
            print(f"ERROR: audio file does not exist: {audio_path}",
                  file=sys.stderr)
            return 2
        print(f"Using user-supplied audio: {audio_path}")
    else:
        audio_path = workdir / "synth.wav"
        print(f"No audio path given — synthesising 10 s test clip at {audio_path}")
        _make_synthetic_clip(audio_path)

    # --- 4. Construct + lazy-load the separator ---
    sep = SourceSeparator(model_name=args.model)
    # Constructor must NOT have loaded the model yet.
    assert sep._model is None, "model loaded too eagerly — should be lazy"
    print("SourceSeparator constructed; model not yet loaded (lazy=True)")

    # --- 5. Run separation for every stem and write each as WAV ---
    written: dict[str, Path] = {}
    for stem in stems:
        print(f"\n--- Separating stem: {stem!r} ---")
        t0 = time.perf_counter()
        try:
            audio, sr = sep.separate(str(audio_path), target_stem=stem)
        except SourceSeparationError as exc:
            print(f"FAILED ({stem}): {exc}", file=sys.stderr)
            return 3
        elapsed = time.perf_counter() - t0
        # Validate the output shape / dtype contract.
        assert audio.dtype == np.float32, audio.dtype
        assert audio.ndim == 1, audio.shape
        assert sr == 44100, sr  # htdemucs native rate
        peak = float(np.max(np.abs(audio)))
        out_path = workdir / f"stem_{stem}.wav"
        sf.write(out_path, audio, sr)
        written[stem] = out_path
        print(f"   ok  — {len(audio):>7d} samples @ {sr} Hz, "
              f"peak={peak:.3f}, took {elapsed:.1f} s, wrote {out_path}")

    # --- 6. Re-run one stem to confirm the cached model is reused ---
    print("\n--- Cached-model re-run (vocals) ---")
    t0 = time.perf_counter()
    sep.separate(str(audio_path), target_stem="vocals")
    elapsed = time.perf_counter() - t0
    print(f"   ok  — second call took {elapsed:.1f} s "
          "(should be similar to first call: model is cached, but "
          "inference itself dominates runtime)")

    # --- 7. Negative paths ---
    print("\n--- Negative-path checks ---")
    try:
        sep.separate(str(audio_path), target_stem="trumpet")
    except SourceSeparationError as exc:
        print(f"   ok  — bad stem rejected: {exc}")
    else:
        print("   FAIL — bad stem name was accepted", file=sys.stderr)
        return 4
    try:
        sep.separate("/no/such/file.mp3", target_stem="vocals")
    except SourceSeparationError as exc:
        print(f"   ok  — missing file rejected: {exc}")
    else:
        print("   FAIL — missing file was accepted", file=sys.stderr)
        return 4

    # --- 8. Summary ---
    print("\n=== ALL CHECKS PASSED ===")
    print(f"Wrote {len(written)} stem files under {workdir}:")
    for stem, path in written.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  {stem:>7s}  {path}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
