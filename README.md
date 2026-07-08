# AutoTabber — Audio-to-Guitar-Tab Transcription

Automatic music transcription: turn an audio recording into beginner-friendly ASCII guitar tablature, using neural pitch detection and source separation. Handles both single-note melodies and chords.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![librosa](https://img.shields.io/badge/librosa-pYIN-4B8BBE?style=flat-square)
![Basic Pitch](https://img.shields.io/badge/Spotify_Basic--Pitch-1DB954?style=flat-square&logo=spotify&logoColor=white)
![Demucs](https://img.shields.io/badge/Meta_Demucs-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)

**[Live demo → Hugging Face Spaces](https://huggingface.co/spaces/shauryadata/autotabberv1)**

## Overview

AutoTabber is an end-to-end **audio ML pipeline** that converts a recording into playable guitar tab. It combines classical DSP (pYIN) with neural models — **Spotify's Basic-Pitch** for polyphonic transcription and **Meta's Demucs** for source separation — and a fretboard-mapping optimizer that picks *playable* finger positions, then renders clean ASCII tab. It runs as a Streamlit app with a persistent SQLite tab history and a live Hugging Face Spaces deployment.

## ML / Signal Pipeline

```
Audio (mp3 / wav / m4a)
   │
   ├─ [optional] Demucs (htdemucs) source separation  → isolate one stem
   │              removes vocals/drums/bass before pitch detection
   ▼
Pitch detection (pluggable)
   ├─ pYIN (librosa)        monophonic, lightweight, cloud-friendly
   └─ Basic-Pitch (neural)  polyphonic / chords (Spotify's CNN)
   │   → NoteEvent(time, midi_note, confidence)
   ▼
TabSimplifier      quantize to a note grid · monophonic collapse · range-simplify
   ▼
FretboardMapper    MIDI → (string, fret), optimized for playability within a
                   difficulty cap (max fret 5 / 7 / 12), not just lowest position
   ▼
TabRenderer        fixed-width ASCII tab with bar lines + note-name legend
   ▼
TabStorage         SQLite history (view / download / delete)
```

## Key Engineering Decisions

- **Pluggable detectors with a deliberate dependency split.** pYIN (pure DSP, ~50 MB) is the default and the only backend that fits the Streamlit Cloud build; the full local install adds Basic-Pitch (TensorFlow) + Demucs (PyTorch, ~1 GB). The app probes for the heavy deps at import and degrades gracefully, so the same codebase runs on cloud and local.
- **Source separation as preprocessing.** On full-band mixes, running Demucs first to isolate the target stem removes the single largest source of transcription noise — far more effective than any downstream filtering.
- **Fretboard mapping as an optimization problem.** The same pitch can be played in several string/fret positions; the mapper chooses assignments that minimize hand movement and stay within a difficulty cap, rather than naively taking the lowest fret.
- **Tunable noise rejection.** Amplitude and minimum-duration thresholds, a monophonic-melody collapse, and a quantization grid let the user trade recall for cleanliness on noisy real-world audio.

## Tech Stack

- **Language:** Python 3.12
- **Audio ML / DSP:** librosa (pYIN), Spotify Basic-Pitch (TensorFlow), Meta Demucs (PyTorch), SciPy/NumPy
- **App / storage:** Streamlit, SQLite, pydub + ffmpeg
- **Testing:** pytest (fretboard mapping, simplification, rendering, chord pipeline, separator)

## Repository Layout

```
app.py                    Streamlit frontend
beginner_tab/             core package
  audio_loader.py         load + resample to mono float32
  pitch_tracker.py        PitchTracker (pYIN) + BasicPitchTracker (neural)
  source_separator.py     Demucs stem isolation (optional, local)
  tab_simplifier.py       quantization + range simplification
  fretboard_mapper.py     playability-optimized string/fret assignment
  tab_renderer.py         ASCII tab rendering
  tab_storage.py          SQLite tab history
tests/                    pytest suite (no audio/model downloads needed)
scripts/                  source-separator smoke test
```

## Running Locally

**Prerequisites:** Python 3.12, and `ffmpeg` for MP3/M4A (`brew install ffmpeg`; WAV works without it).

```bash
python3 -m venv venv && source venv/bin/activate

# Cloud-style / lightweight — pYIN only, no PyTorch/TensorFlow (~50 MB)
pip install -r requirements.txt

# Full local — adds Basic-Pitch (polyphonic) + Demucs (source separation, ~1 GB)
pip install -r requirements-local.txt

streamlit run app.py      # http://localhost:8501
pytest tests/ -v          # run the test suite
```

## Limitations

Tabs capture note sequence, not rhythm durations; detection accuracy depends on recording quality and complexity; the fretboard mapper optimizes for hand comfort over musical phrasing. It's a learning aid, not a professional transcription tool. Upload only audio you own or are authorized to use.

## Authors

Built by **Shauryaditya Singh**.

## License

MIT — see [LICENSE](LICENSE).
