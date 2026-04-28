# AutoTabber

Convert audio recordings into beginner-friendly ASCII guitar tablature.
Supports both single-note melodies and chords.

**Live demo:** [autotabber.streamlit.app](https://autotabber.streamlit.app)

---

## Features

| Feature | Detail |
|---|---|
| Audio input | MP3, WAV, M4A |
| pYIN detector | Monophonic pitch detection via librosa (single notes) |
| Basic-pitch detector | Polyphonic AI pitch detection via Spotify's neural network (chords) |
| Source Separation | Optional Demucs preprocessing — isolates one stem (vocals/drums/bass/other) before pitch detection for dramatically cleaner tabs (local only) |
| Monophonic Melody Mode | Collapses near-simultaneous detections into a clean single-line melody |
| Note Filtering | User-tunable amplitude and minimum-duration sliders to reject noise / artifacts |
| Difficulty Mode | Beginner (max fret 5) · Intermediate (max fret 7) · Advanced (max fret 12) |
| One-string mode | High-e only for absolute beginners |
| Note grid | Quarter / 8th / 16th note quantisation |
| Tab history | SQLite-backed storage with view, download, and delete |
| Output | ASCII tab with bar lines every 16 notes, fixed-width columns, note-name legend; downloadable as `.txt` |

---

## Requirements

### Python

Python **3.12** is required.

### ffmpeg (for MP3 / M4A)

pydub delegates audio decoding to **ffmpeg**. WAV files work without it.

| Platform | Install command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Windows | Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add `bin/` to PATH |

Verify: `ffmpeg -version`

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/shauryadata/autotabberv1.git
cd autotabberv1

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**3. Install dependencies — pick the install that matches how you'll use AutoTabber:**

| Install | What you get | Command |
|---|---|---|
| **Cloud-style / lightweight** | pYIN detector only · no PyTorch · no Demucs · ~50 MB of dependencies — matches the Streamlit Cloud build exactly | `pip install -r requirements.txt` |
| **Full local** | Everything above **plus** Basic-pitch (polyphonic AI) **and** Demucs source separation · pulls in PyTorch with a pinned `torch>=2.2,<2.6` · ~1 GB of dependencies | `pip install -r requirements-local.txt` |

```bash
# 4. Launch the app
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

> **What's available where:** Basic-pitch needs TensorFlow and Demucs needs
> PyTorch — neither has wheels that fit the Streamlit Cloud Python build, so
> the cloud deployment runs **pYIN only with no source separation**. The
> full local install enables every feature: Basic-pitch chord detection
> (selected by default) and the optional Demucs preprocessing stage.

---

## Usage

1. **Upload** an MP3, WAV, or M4A audio file.
2. **Choose a detector** in the sidebar:
   - *Basic-pitch* (recommended, **local only**) — chords and full mixes.
   - *pYIN* — clean single-note melodies; the only option on Streamlit Cloud.
3. **Adjust settings** (every control has a tooltip in the sidebar):
   - **Note Filtering** — two sliders that prune noisy / artifact notes:
     - *Amplitude Threshold* (0.10–0.80, default 0.40) drops detections
       quieter than the threshold.
     - *Min Note Duration (ms)* (30–200, default 60) drops detections
       shorter than the threshold.
   - **Monophonic Melody Mode** *(checkbox, default on)* — keeps only the
     loudest note in each 50 ms window. Produces a clean single-line tab
     even from polyphonic input. Turn off if you want Basic-pitch to render
     full chords.
   - **Difficulty Mode** *(select slider)* —
     - *Beginner (max fret 5)* — easy open-position playability;
     - *Intermediate (max fret 7)* — covers most beginner songs;
     - *Advanced (max fret 12)* — opens the full first 12 frets.
   - **Note Grid** — quantisation resolution (quarter / 8th / 16th notes).
   - **One-String Mode** — high-e string only, for absolute beginners.
   - **Columns per tab line** — line width in 16-note (one-measure) steps.
   - **Advanced expanders** — per-detector tuning (HPSS, onset / frame
     thresholds, etc.) for users who want finer control.
4. Click **Generate Tab**.
5. View the tab on screen, **download** as `.txt`, or find it later in
   **Tab History**.

### Reading the tab

```
e|-3-----5-----3-----0----|---8-----7-----3-----3--|--5--3--10---|
B|------------------------|------------------------|-------------|
G|------------------------|------------------------|-------------|
D|------------------------|------------------------|-------------|
A|------------------------|------------------------|-------------|
E|------------------------|------------------------|-------------|

Notes:  G4  A4  G4  E4  C5  B4  G4  G4  A4  G4  D5
```

- Row labels (`e B G D A E`) are guitar strings, high to low.
- **Numbers** = fret to press. **0** = open string. Each column is a fixed
  3 characters wide so single- and double-digit frets stay aligned.
- **Dashes** = string not played on that beat.
- **Bar lines** (`|`) appear after every 16 columns — one measure of 16th
  notes — to make rhythm groupings easy to read.

---

## Quality Tips

AutoTabber is only as good as the audio you feed it. A few minutes of source
prep will dramatically improve the resulting tab:

- **Upload clean, isolated guitar audio.** A solo guitar track or a
  transcription stem produces the cleanest results. Full mixes with vocals,
  drums, and bass leak into pitch detection and add noise.
- **Trim to the section you want.** Generate tabs for the riff, verse, or
  solo you actually want — not the whole song. Shorter clips run faster and
  detect more accurately.
- **Prefer lossless or high-bitrate sources.** WAV and high-bitrate MP3 /
  M4A preserve the harmonic detail Basic-pitch and pYIN need. Heavily
  compressed clips lose pitch information and produce more artifacts.
- **Tune the Note Filtering sliders if you see junk notes.** Raise
  *Amplitude Threshold* to 0.5–0.6 to drop bleed; raise *Min Note Duration*
  to 80–100 ms to drop pick-attack blips. Drop them again if real notes are
  being filtered out.
- **Start in Beginner Difficulty Mode.** It caps the tab at fret 5 so every
  position is in open-position territory — easy to reach without shifting
  your hand. Move up to Intermediate (fret 7) or Advanced (fret 12) once
  you're comfortable.
- **Leave Monophonic Melody Mode on for melody work.** It guarantees one
  note per beat. Turn it off only when you specifically want Basic-pitch to
  render chord stacks.
- **For best results on full-band recordings, enable Source Separation**
  and choose the stem that matches your target instrument — *Other* for
  guitar / keys, *Vocals* for the sung melody. The 30–90 s extra
  processing typically removes more transcription error than any amount
  of slider tuning. Local installs only — see the next section.

---

## Source Separation Mode (Local Only)

When the upload is a full-band recording, the cleanest path to a usable
tab is to first **isolate the instrument you actually care about** and
only then run pitch detection on it. AutoTabber wires up Spotify's
[Demucs](https://github.com/facebookresearch/demucs) htdemucs model
behind a single sidebar checkbox — when enabled, every Generate Tab
click first runs the audio through Demucs and feeds only the chosen
stem into the rest of the pipeline. Vocals, drums, and bass disappear
from the input that the pitch tracker actually sees, which removes the
single largest source of transcription noise on real-world recordings.

**Pick the stem that matches what you want to play:**

| Stem | When to pick it |
|---|---|
| **Other (guitar/keys)** — *recommended for guitar tabs* | Every guitar, piano, synth, or non-percussive instrument that isn't a lead vocal. This is the right choice for ~90 % of guitar transcription work. |
| **Vocals (sing the melody)** | When you want to learn the sung lead line on guitar — useful for melody studies and sing-along tabs. |
| **Bass** | Bass-line transcription. Pair with Beginner Difficulty Mode and the low strings. |
| **Drums** *(not recommended)* | Drums are not pitched, so AutoTabber will produce noise. Listed for completeness; almost never useful. |

**Cost:** Source separation adds **30–90 seconds per song** on CPU
(faster with a GPU). The Demucs model also downloads ~80 MB of weights
on first use — you'll see a progress bar in the terminal, then the
weights are cached for the rest of your sessions.

**Availability:** Source separation requires the full local install
(`pip install -r requirements-local.txt`), which pulls in PyTorch and
Demucs. **The Streamlit Cloud deployment does not include these
dependencies** — the sidebar will show an info message explaining
that the cloud build runs on the full mix only. Run AutoTabber locally
to use this feature.

---

## Project Layout

```
AutoTabber/
├── app.py                          # Streamlit frontend
├── requirements.txt                # Cloud deployment dependencies
├── requirements-local.txt          # Full local dependencies (with Basic-pitch)
├── packages.txt                    # System packages for Streamlit Cloud
├── README.md
├── beginner_tab/                   # Core package
│   ├── __init__.py                 # Package init + Basic-pitch / Demucs probes
│   ├── audio_loader.py             # AudioLoader class
│   ├── pitch_tracker.py            # PitchTracker (pYIN) and BasicPitchTracker classes
│   ├── tab_simplifier.py           # TabSimplifier class
│   ├── fretboard_mapper.py         # FretboardMapper class
│   ├── tab_renderer.py             # TabRenderer class
│   ├── tab_storage.py              # TabStorage class (SQLite, with stem column)
│   └── source_separator.py         # SourceSeparator (Demucs, optional, local-only)
├── scripts/
│   └── verify_source_separator.py  # Standalone smoke-test for the separator
└── tests/
    ├── __init__.py
    ├── test_frestboard_mapper.py     # FretboardMapper scoring & placement
    ├── test_tab_renderer.py          # ASCII rendering, bar lines, widths
    ├── test_tab_simplifier.py        # Quantisation & range simplification
    ├── test_chord_pipeline.py        # Chord pipeline + cross-pipeline tests
    └── test_source_separator.py      # Demucs separator (skips when unavailable)
```

---

## Running the Tests

```bash
pytest tests/ -v
```

Tests cover FretboardMapper, TabSimplifier, TabRenderer, chord pipeline,
and TabStorage without requiring audio files or model downloads.

---

## Architecture

```
AudioLoader
    |  (mono float32 array, sample_rate)
    v
PitchTracker / BasicPitchTracker
    |  list[NoteEvent]  (time, midi_note, confidence)
    v
TabSimplifier
    |  list[QuantizedNote] or list[ChordNote]
    v
FretboardMapper
    |  list[TabNote] or list[ChordTabNote]
    v
TabRenderer
    |  str (ASCII guitar tab)
    v
TabStorage
       SQLite database (tabs.db)
```

---

## Known Limitations

- Basic-pitch requires Python 3.12 or earlier (TensorFlow dependency).
- Generated tabs show note sequence only, not rhythm durations.
- Detection accuracy depends on recording quality and complexity.
- The fretboard mapper optimizes for hand comfort, not musical phrasing.
- Long files take longer to process; trim to the section you need.
- AutoTabber is a learning aid, not a professional transcription tool.

---

## Legal

Upload only audio you own or are authorized to use. The YouTube URL field
is for labelling only. AutoTabber does not download from YouTube.
