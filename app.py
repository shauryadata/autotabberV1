# ==========================================================================
# File:        app.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Streamlit web application frontend for AutoTabber.  Provides
#              an interactive GUI where users upload audio files, choose a
#              pitch-detection backend, configure beginner-friendly settings,
#              and receive a downloadable ASCII guitar tab.  Also includes
#              a persistent tab history backed by SQLite.
#
# Run with:    streamlit run app.py
# Requires:    ffmpeg on PATH for MP3 / M4A conversion (WAV works without it).
# ==========================================================================

"""AutoTabber: Streamlit frontend.

Run with: streamlit run app.py

Requires ffmpeg on PATH for MP3 / M4A conversion (WAV works without it).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# Import all pipeline classes and exceptions from the beginner_tab package
from beginner_tab import (
    AudioLoader, AudioLoadError,
    BasicPitchTracker, PitchTracker, PitchDetectionError,
    TabSimplifier, FretboardMapper, TabRenderer,
    TabStorage, TabStorageError,
    BASIC_PITCH_AVAILABLE, BASIC_PITCH_UNAVAILABLE_REASON,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoTabber",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared storage ────────────────────────────────────────────────────────────
@st.cache_resource
def get_storage() -> TabStorage:
    """Create or retrieve the singleton TabStorage instance.

    Uses Streamlit's ``cache_resource`` decorator so the database
    connection is created only once per app session, not on every rerun.

    Returns:
        TabStorage: A ready-to-use storage instance backed by ``tabs.db``.
    """
    return TabStorage("tabs.db")

# Singleton database connection shared across all reruns
storage = get_storage()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Settings")

# 1) Detector choice — radio button
st.sidebar.subheader("1. Pitch Detector")
if BASIC_PITCH_AVAILABLE:
    # Both detectors available — let the user choose
    detector = st.sidebar.radio(
        "Choose a detector",
        options=["Basic-pitch (Polyphonic AI — recommended)", "pYIN (Monophonic, fast)"],
        index=0,
        help=(
            "Basic-pitch: Spotify neural network — detects chords, works on "
            "full-mix recordings.  First run loads the model (~5-10 s).\n\n"
            "pYIN: Classic algorithm — single note at a time, no model download."
        ),
    )
else:
    # basic-pitch is not available — show context-aware guidance.
    detector = "pYIN (Monophonic, fast)"
    # Detect whether we are on Streamlit Cloud or running locally
    _on_streamlit_cloud = os.environ.get("STREAMLIT_SHARING_MODE") or \
        os.path.exists("/mount/src")
    if _on_streamlit_cloud:
        st.sidebar.info(
            "Running in cloud mode — pYIN detector active. "
            "For AI-powered Basic-pitch detection, run the app locally."
        )
    else:
        st.sidebar.warning(
            "Basic-pitch detector is not available.\n\n"
            + BASIC_PITCH_UNAVAILABLE_REASON
        )
# Boolean flag controlling which pipeline branch (polyphonic vs monophonic) runs
use_basic_pitch = detector.startswith("Basic")

st.sidebar.divider()

# Detector-specific tuning (advanced)
if use_basic_pitch:
    with st.sidebar.expander("Advanced: Basic-pitch tuning"):
        onset_threshold: float = st.slider(
            "Onset Threshold", min_value=0.20, max_value=0.90, value=0.50, step=0.05,
            help="Confidence to start a new note.  Lower = more notes (noisier).",
        )
        frame_threshold: float = st.slider(
            "Frame Threshold", min_value=0.10, max_value=0.80, value=0.30, step=0.05,
            help="Confidence to sustain a note.  Keep below Onset Threshold.",
        )
        min_note_ms: float = st.slider(
            "Min Note Length (ms)", min_value=40, max_value=300, value=80, step=20,
            help="Shortest note kept.  80 ms filters most spurious blips.",
        )
else:
    with st.sidebar.expander("Advanced: pYIN tuning"):
        use_hpss: bool = st.checkbox(
            "HPSS pre-processing", value=True,
            help="Harmonic-Percussive separation before pYIN.",
        )
        min_confidence: float = st.slider(
            "Pitch Confidence", min_value=0.30, max_value=0.95, value=0.45, step=0.05,
            help="Minimum pYIN voiced-probability to keep a frame.",
        )

# Aggressive note-cleaning filters — apply to BOTH pitch trackers AFTER
# raw detection.  These two sliders are the most useful tuning knobs once a
# user understands what gets dropped: amplitude removes background noise /
# bleed, min-duration removes pick-attack blips and detector artifacts.
st.sidebar.subheader("Note Filtering")
amplitude_threshold: float = st.sidebar.slider(
    "Amplitude Threshold",
    min_value=0.10, max_value=0.80, value=0.40, step=0.05,
    help=(
        "Drop any detected note quieter than this.  Higher = stricter "
        "(removes background noise and instrument bleed).  Lower = keeps "
        "more notes (including artifacts)."
    ),
)
min_note_duration_ms: int = st.sidebar.slider(
    "Min Note Duration (ms)",
    min_value=30, max_value=200, value=60, step=10,
    help=(
        "Drop any detected note shorter than this.  60 ms is the shortest "
        "intentionally-played guitar note; below that you are almost "
        "always looking at pick-attack transients or detector artifacts."
    ),
)
# Convert ms → seconds for the tracker constructors below.
min_duration_s: float = min_note_duration_ms / 1000.0

st.sidebar.divider()

# Monophonic-melody filter — applies to BOTH pitch trackers.  When enabled,
# any cluster of near-simultaneous detected notes (within a 50 ms window) is
# collapsed to the single loudest note, producing a clean single-line melody
# even on polyphonic recordings.  Recommended for beginner tabs.
monophonic_mode: bool = st.sidebar.checkbox(
    "Monophonic Melody Mode (recommended)",
    value=True,
    help=(
        "Keep only the loudest note in each 50 ms window so the tab is a "
        "clean single-line melody.  Turn off to render full chords from "
        "Basic-pitch."
    ),
)

st.sidebar.divider()

# 2) Difficulty Mode — discrete select_slider mapping each difficulty tier
# to a max-fret cap.  Replaces the previous free-form integer slider so
# users pick a meaningful skill level instead of guessing a fret number.
st.sidebar.subheader("2. Difficulty Mode")
DIFFICULTY_MAX_FRETS: dict[str, int] = {
    "Beginner (max fret 5)": 5,
    "Intermediate (max fret 7)": 7,
    "Advanced (max fret 12)": 12,
}
difficulty_label: str = st.sidebar.select_slider(
    "Difficulty Mode",
    options=list(DIFFICULTY_MAX_FRETS.keys()),
    value="Beginner (max fret 5)",
    help=(
        "Highest fret the tab is allowed to use.  Beginner stays in the "
        "first 5 frets (open-position chords / easy melodies); Intermediate "
        "reaches fret 7 (covers most beginner songs); Advanced opens up the "
        "full first 12 frets."
    ),
)
# Resolve the chosen tier to the integer cap that the rest of the
# pipeline (FretboardMapper, renderer, storage) already understands.
max_fret: int = DIFFICULTY_MAX_FRETS[difficulty_label]

# 3) Note Grid — selectbox
st.sidebar.subheader("3. Note Grid")
subdivision: int = st.sidebar.selectbox(
    "Quantisation resolution",
    options=[1, 2, 4], index=1,
    format_func=lambda v: {1: "Quarter notes", 2: "8th notes", 4: "16th notes"}[v],
)

st.sidebar.divider()

one_string_mode: bool = st.sidebar.checkbox(
    "One-String Mode (high e only)", value=False,
    help="Play everything on the high-e string.",
)
notes_per_line: int = st.sidebar.slider(
    "Columns per tab line",
    min_value=16, max_value=64, value=48, step=16,
    help=(
        "Tab columns per line.  Stay on multiples of 16 so each line "
        "ends on a measure boundary (one measure = 16 sixteenth-note "
        "columns).  48 = 3 measures per line, matching the standard "
        "beginner-tab layout."
    ),
)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("AutoTabber")
st.subheader("Upload a song and get beginner guitar tabs instantly")

# How-to guide (collapsible)
with st.expander("How to Use AutoTabber", expanded=False):
    st.markdown(
        """
**Step 1:** Upload an MP3, WAV, or M4A file using the uploader below.

**Step 2:** Choose your pitch detector in the sidebar
(Basic-pitch is recommended for most songs).

**Step 3:** Adjust the **Max Fret** slider to limit tab difficulty
(5 = beginner, 12 = advanced).

**Step 4:** Click **Generate Tab** and wait for processing.

**Step 5:** View your tab, download it as `.txt`, or find it
later in the Tab History section.
        """
    )

# 4) File uploader
col_up, col_yt = st.columns([3, 2])
with col_up:
    uploaded_file = st.file_uploader(
        "Upload Audio File (MP3 / WAV / M4A)", type=["mp3", "wav", "m4a"],
        help="MP3 and M4A files require ffmpeg to be installed on your system.",
    )
with col_yt:
    youtube_link = st.text_input(
        "YouTube URL (reference / label only)",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    if youtube_link.strip():
        st.warning(
            "**Disclaimer**: For labelling only.  AutoTabber does NOT download "
            "from YouTube.  Upload only audio you own or are authorised to use."
        )

st.divider()

# Maximum upload size in megabytes — prevents excessive memory usage
MAX_FILE_MB = 50

if uploaded_file is None:
    st.info("Upload an audio file above, then press **Generate Tab**.")
else:
    file_mb = len(uploaded_file.getvalue()) / 1_048_576
    if file_mb > MAX_FILE_MB:
        st.error(f"File is {file_mb:.1f} MB — max {MAX_FILE_MB} MB.  Please trim it.")
        st.stop()

    st.success(f"Ready: **{uploaded_file.name}** ({file_mb:.1f} MB)")
    # 5) Generate button
    generate = st.button("Generate Tab", type="primary", use_container_width=True)

    if generate:
        # Save uploaded bytes to a temp file so AudioLoader can read it by path
        suffix = Path(uploaded_file.name).suffix.lower()
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(uploaded_file.getvalue())

            with st.status("Processing…", expanded=True) as status:

                # 1. Load
                st.write("**1/4** Loading audio…")
                try:
                    audio, sr = AudioLoader(tmp_path).load()
                except AudioLoadError as exc:
                    st.error(f"Audio load failed: {exc}")
                    st.stop()

                # Audio duration in seconds — used for user warnings
                duration = len(audio) / sr
                st.write(f"Loaded — {duration:.1f} s · {sr} Hz")
                if np.max(np.abs(audio)) < 1e-5:
                    st.error("Audio appears silent.")
                    st.stop()
                if duration < 0.5:
                    st.warning("Very short (<0.5 s) — results may be unreliable.")
                if duration > 240:
                    st.warning(f"{duration:.0f} s file — may take a while.")

                # 2. Detect
                if use_basic_pitch:
                    st.write("**2/4** Running Basic-pitch neural detector…")
                    st.caption("First run loads the TF model — allow 5-10 s extra.")
                    try:
                        tracker = BasicPitchTracker(
                            audio, sr,
                            onset_threshold=onset_threshold,
                            frame_threshold=frame_threshold,
                            minimum_note_length_ms=min_note_ms,
                            monophonic_mode=monophonic_mode,
                            amplitude_threshold=amplitude_threshold,
                            min_duration_s=min_duration_s,
                        )
                        raw_notes = tracker.track()
                        tempo = tracker.estimate_tempo()
                    except PitchDetectionError as exc:
                        st.error(f"Basic-pitch failed: {exc}")
                        st.stop()
                else:
                    st.write("**2/4** Running pYIN pitch detection…")
                    try:
                        tracker = PitchTracker(
                            audio, sr,
                            min_confidence=min_confidence,
                            use_hpss=use_hpss,
                            monophonic_mode=monophonic_mode,
                            amplitude_threshold=amplitude_threshold,
                            min_duration_s=min_duration_s,
                        )
                        raw_notes = tracker.track()
                        tempo = tracker.estimate_tempo()
                    except PitchDetectionError as exc:
                        st.error(f"pYIN failed: {exc}")
                        st.stop()

                if not raw_notes:
                    st.error(
                        "No notes detected.  " + (
                            "Lower the Onset Threshold." if use_basic_pitch
                            else "Lower Pitch Confidence or enable HPSS."
                        )
                    )
                    st.stop()

                st.write(
                    f"Detected **{len(raw_notes)}** events · "
                    f"Tempo ≈ **{tempo:.0f} BPM**"
                )

                # 3. Quantise
                st.write("**3/4** Quantising to beat grid…")
                simplifier = TabSimplifier(raw_notes, tempo=tempo, subdivision=subdivision)
                if use_basic_pitch:
                    quantized = simplifier.quantize_chords()
                    quantized = simplifier.simplify_range_chords(quantized)
                else:
                    quantized = simplifier.quantize()
                    quantized = simplifier.simplify_range(quantized)
                st.write(
                    f"{len(quantized)} "
                    f"{'chord slots' if use_basic_pitch else 'notes'} after quantisation"
                )

                # 4. Map
                st.write("**4/4** Mapping to fretboard…")
                mapper = FretboardMapper(
                    max_fret_limit=max_fret, one_string_mode=one_string_mode
                )
                tab_data = (
                    mapper.map_chords(quantized) if use_basic_pitch else mapper.map(quantized)
                )
                if mapper.skipped_count:
                    st.warning(
                        f"{mapper.skipped_count} note(s) skipped (beyond fret {max_fret}).  "
                        "Increase Max Fret to include them."
                    )
                if not tab_data:
                    st.error("No notes placed — increase Max Fret or disable One-String Mode.")
                    st.stop()
                st.write(f"{len(tab_data)} events placed on fretboard.")

                # Render
                renderer = TabRenderer(notes_per_line=notes_per_line)
                detector_label = "basic-pitch" if use_basic_pitch else "pyin"
                tab_text = (
                    renderer.render_chords(
                        tab_data, tempo=tempo, max_fret=max_fret,
                        one_string_mode=one_string_mode, duration=duration,
                    )
                    if use_basic_pitch
                    else renderer.render(
                        tab_data, tempo=tempo, max_fret=max_fret,
                        one_string_mode=one_string_mode, duration=duration,
                    )
                )

                # Auto-save
                try:
                    storage.save(
                        filename=uploaded_file.name, tab_text=tab_text,
                        reference_url=youtube_link.strip() or None,
                        detector=detector_label, tempo=tempo,
                        max_fret=max_fret, one_string=one_string_mode,
                        note_count=len(tab_data),
                    )
                except TabStorageError as exc:
                    st.warning(f"Tab generated but could not be saved to history: {exc}")

                status.update(label="Tab generated!", state="complete")

            st.success(
                f"Tab generated successfully — {len(tab_data)} events, "
                f"{tempo:.0f} BPM, max fret {max_fret}."
            )
            st.subheader("Generated Guitar Tab")
            st.code(tab_text, language=None)

            # Metadata header prepended to the downloadable .txt file
            meta = [
                "AutoTabber — Generated Guitar Tab",
                f"Source file : {uploaded_file.name}",
            ]
            if youtube_link.strip():
                meta.append(f"Reference   : {youtube_link.strip()}")
            meta += [
                f"Detector    : {detector_label}",
                f"Tempo       : {tempo:.0f} BPM",
                f"Max fret    : {max_fret}",
                f"One-string  : {'yes' if one_string_mode else 'no'}",
                "-" * 44, "",
            ]
            # 6) Download button
            st.download_button(
                "Download Tab (.txt)",
                data="\n".join(meta) + "\n" + tab_text,
                file_name=f"{Path(uploaded_file.name).stem}_tab.txt",
                mime="text/plain",
                use_container_width=True,
            )

        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            with st.expander("Traceback"):
                st.exception(exc)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# ── Tab History ───────────────────────────────────────────────────────────────
st.divider()
with st.expander(f"Tab History ({storage.count()} saved)", expanded=False):
    records = storage.list_tabs()
    if not records:
        st.info("No tabs saved yet — generate one above!")
    else:
        for rec in records:
            col_info, col_btn = st.columns([5, 1])
            with col_info:
                one_lbl = " · one-string" if rec["one_string"] else ""
                st.markdown(
                    f"**#{rec['id']}** {rec['filename']} "
                    f"— {rec['detector']} · {rec['tempo']:.0f} BPM · "
                    f"fret {rec['max_fret']}{one_lbl} · "
                    f"{rec['note_count']} events · {rec['created_at']} UTC"
                )
            with col_btn:
                if st.button("View / \u2193", key=f"view_{rec['id']}"):
                    st.session_state["viewing_tab_id"] = rec["id"]

        viewing_id = st.session_state.get("viewing_tab_id")
        if viewing_id:
            full = storage.get_tab(viewing_id)
            if full:
                st.subheader(f"Tab #{viewing_id} — {full['filename']}")
                st.code(full["tab_text"], language=None)
                col_dl, col_del = st.columns([3, 1])
                with col_dl:
                    st.download_button(
                        "Download (.txt)", data=full["tab_text"],
                        file_name=f"tab_{viewing_id}.txt", mime="text/plain",
                        key=f"dl_{viewing_id}",
                    )
                with col_del:
                    if st.button("Delete", key=f"del_{viewing_id}", type="secondary"):
                        storage.delete_tab(viewing_id)
                        st.session_state.pop("viewing_tab_id", None)
                        st.rerun()

# ── Limitations ───────────────────────────────────────────────────────────────
with st.expander("Known Limitations & Tips", expanded=False):
    st.markdown(
        """
**Basic-pitch (recommended)**
- Detects chords; works on full-mix recordings.
- First run loads the TensorFlow model (~5-10 s overhead).
- Tune **Onset Threshold**: lower = more notes, higher = more selective.

**pYIN (classic)**
- Single-note melodies only.  Fast, no model download.  Enable HPSS for mixed tracks.

**Tab format**
- Note sequence only — rhythm durations are not encoded.
- Fret positions minimise hand movement, not musical phrasing.

**ffmpeg (MP3 / M4A)**
- macOS: `brew install ffmpeg` · Ubuntu: `sudo apt install ffmpeg`
- Windows: ffmpeg.org, add `bin/` folder to PATH.

**Accuracy** — AutoTabber is a learning aid.  Always verify by ear.
        """
    )
