# ==========================================================================
# File:        tab_renderer.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Renders mapped guitar tab notes as human-readable ASCII
#              tablature strings.  Supports monophonic (one note per column)
#              and polyphonic (chord columns with multiple active strings).
#              Produces the standard 6-line EADGBE format with a header,
#              note-name legend, and automatic line wrapping.
# ==========================================================================

"""Render guitar tab notes as ASCII tab strings.

Two render paths share the same header/footer logic:

render()        — monophonic: one active string per column.
render_chords() — polyphonic: multiple active strings per column (chords).

Both produce the same 6-line EADGBE format.
"""

from __future__ import annotations

from .fretboard_mapper import TabNote, ChordTabNote, STRING_NAMES

# ---------------------------------------------------------------------------
# Note-name helper (no librosa dependency)
# ---------------------------------------------------------------------------
# The twelve chromatic pitch classes in order, used to convert MIDI → name
_CHROMATIC: list[str] = ["C", "C#", "D", "D#", "E", "F",
                          "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_note_name(midi: int) -> str:
    """Convert a MIDI note number to a human-readable note name.

    Uses the standard formula: pitch class = midi mod 12, octave = midi // 12 - 1.

    Args:
        midi: MIDI note number (0–127).

    Returns:
        str: Note name with octave, e.g. 64 → ``'E4'``, 69 → ``'A4'``.
    """
    return f"{_CHROMATIC[midi % 12]}{(midi // 12) - 1}"


# ---------------------------------------------------------------------------
# Column descriptor types used internally by the renderer
# ---------------------------------------------------------------------------
# Monophonic column: (active_string_idx, fret_str, column_width_in_chars)
_MonoCol = tuple[int, str, int]

# Polyphonic column: ({string_idx: fret_str}, column_width_in_chars)
_ChordCol = tuple[dict[int, str], int]


class TabRenderer:
    """Render tab notes as a human-readable ASCII guitar tab.

    Example::

        r = TabRenderer(notes_per_line=16)
        print(r.render(tab_notes))           # monophonic
        print(r.render_chords(chord_notes))  # polyphonic
    """

    DEFAULT_NOTES_PER_LINE: int = 16

    def __init__(self, notes_per_line: int = DEFAULT_NOTES_PER_LINE) -> None:
        """Initialise TabRenderer.

        Args:
            notes_per_line: Note columns per line before wrapping.
        """
        self.notes_per_line = notes_per_line

    # ------------------------------------------------------------------
    # Monophonic render
    # ------------------------------------------------------------------

    def render(
        self,
        tab_notes: list[TabNote],
        tempo: float | None = None,
        max_fret: int | None = None,
        one_string_mode: bool = False,
    ) -> str:
        """Render monophonic tab notes as ASCII guitar tab.

        Args:
            tab_notes: Output of :meth:`FretboardMapper.map`.
            tempo: Optional BPM for the header.
            max_fret: Optional max-fret setting for the header.
            one_string_mode: Whether one-string mode was active.

        Returns:
            Multi-line ASCII string.
        """
        if not tab_notes:
            return (
                "No notes could be mapped to the fretboard.\n"
                "Try increasing the max-fret setting or lowering the "
                "pitch-detection confidence threshold."
            )

        columns = self._build_mono_columns(tab_notes)
        note_names = [_midi_to_note_name(midi) for _, _, _, midi in tab_notes]
        return self._render_body(
            columns, note_names, tempo, max_fret, one_string_mode, polyphonic=False
        )

    def _build_mono_columns(self, tab_notes: list[TabNote]) -> list[_MonoCol]:
        cols: list[_MonoCol] = []
        for _, string_idx, fret, _ in tab_notes:
            fret_str = str(fret)
            width = max(len(fret_str) + 2, 3)
            cols.append((string_idx, fret_str, width))
        return cols

    # ------------------------------------------------------------------
    # Polyphonic / chord render
    # ------------------------------------------------------------------

    def render_chords(
        self,
        chord_notes: list[ChordTabNote],
        tempo: float | None = None,
        max_fret: int | None = None,
        one_string_mode: bool = False,
    ) -> str:
        """Render polyphonic chord notes as ASCII guitar tab.

        Columns with multiple active strings display all fret numbers
        simultaneously — this represents a chord.

        Args:
            chord_notes: Output of :meth:`FretboardMapper.map_chords`.
            tempo: Optional BPM for the header.
            max_fret: Optional max-fret setting for the header.
            one_string_mode: Whether one-string mode was active.

        Returns:
            Multi-line ASCII string.
        """
        if not chord_notes:
            return (
                "No chords could be mapped to the fretboard.\n"
                "Try increasing the max-fret setting or adjusting the "
                "onset threshold."
            )

        columns = self._build_chord_columns(chord_notes)
        # For the note legend: show all MIDI values, comma-joined for chords
        note_names: list[str] = []
        for _, positions, midis in chord_notes:
            if len(midis) == 1:
                note_names.append(_midi_to_note_name(midis[0]))
            else:
                note_names.append("(" + "+".join(_midi_to_note_name(m) for m in midis) + ")")

        return self._render_body(
            columns, note_names, tempo, max_fret, one_string_mode, polyphonic=True
        )

    def _build_chord_columns(self, chord_notes: list[ChordTabNote]) -> list[_ChordCol]:
        cols: list[_ChordCol] = []
        for _, positions, _ in chord_notes:
            fret_map: dict[int, str] = {s: str(f) for s, f in positions}
            max_len = max((len(v) for v in fret_map.values()), default=1)
            width = max(max_len + 2, 3)
            cols.append((fret_map, width))
        return cols

    # ------------------------------------------------------------------
    # Shared rendering core
    # ------------------------------------------------------------------

    def _render_body(
        self,
        columns: list[_MonoCol] | list[_ChordCol],
        note_names: list[str],
        tempo: float | None,
        max_fret: int | None,
        one_string_mode: bool,
        polyphonic: bool,
    ) -> str:
        """Build the complete ASCII tab string from column descriptors.

        This method is shared by both :meth:`render` and :meth:`render_chords`.
        It handles the header, line-wrapped tab body, and note-name legend.

        Args:
            columns: List of column descriptors (_MonoCol or _ChordCol).
            note_names: Human-readable note labels (e.g. ``'E4'``, ``'(E4+G4)'``).
            tempo: Song tempo in BPM, shown in the header.
            max_fret: Max-fret setting, shown in the header.
            one_string_mode: Whether one-string mode was active.
            polyphonic: True for chord columns, False for monophonic columns.

        Returns:
            str: The complete multi-line ASCII guitar tab.
        """
        # Split columns into chunks that fit on one line of tab
        chunks = [
            columns[i : i + self.notes_per_line]
            for i in range(0, len(columns), self.notes_per_line)
        ]

        parts: list[str] = []

        # --- Header section ---
        parts.append("=" * 44)
        parts.append("  AutoTabber — Beginner Guitar Tab")
        parts.append("=" * 44)
        # Metadata line: tuning, tempo, settings
        meta: list[str] = ["Standard tuning (EADGBE)"]
        if tempo is not None:
            meta.append(f"Tempo ≈ {tempo:.0f} BPM")
        if max_fret is not None:
            meta.append(f"Max fret: {max_fret}")
        if one_string_mode:
            meta.append("One-string mode")
        if polyphonic:
            meta.append("Polyphonic (Basic-pitch)")
        parts.append("  " + " | ".join(meta))
        parts.append("")

        # --- Tab body: 6 string rows per chunk ---
        for chunk in chunks:
            for str_idx, name in enumerate(STRING_NAMES):
                row = f"{name}|"  # e.g. "e|" or "E|"
                for col in chunk:
                    if polyphonic:
                        # Chord column: fret_map has entries only for active strings
                        fret_map, width = col  # type: ignore[misc]
                        fret_str = fret_map.get(str_idx)  # type: ignore[union-attr]
                        if fret_str is not None:
                            # This string is active in the chord — show fret number
                            row += fret_str.center(width, "-")
                        else:
                            # Inactive string — fill with dashes
                            row += "-" * width
                    else:
                        # Mono column: only one string is active per column
                        active_str, fret_str, width = col  # type: ignore[misc]
                        if str_idx == active_str:
                            row += fret_str.center(width, "-")
                        else:
                            row += "-" * width
                row += "|"
                parts.append(row)
            parts.append("")  # blank line between chunks

        # --- Note-name legend ---
        parts.append("Notes:  " + "  ".join(note_names))
        parts.append("")
        parts.append(
            "How to read: numbers = fret, 0 = open string, "
            "- = string not played, (X+Y) = chord."
        )
        return "\n".join(parts)
