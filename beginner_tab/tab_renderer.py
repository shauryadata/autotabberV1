# ==========================================================================
# File:        tab_renderer.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Renders mapped guitar tab notes as human-readable ASCII
#              tablature.  Supports monophonic (one note per column) and
#              polyphonic (chord columns with multiple active strings) input.
#              Produces the standard 6-line EADGBE format with:
#                * a stats header (tempo / note count / duration),
#                * a settings line (tuning / max fret / mode),
#                * a tab body with fixed 3-char columns and bar lines every
#                  16 columns (= one measure of 16th notes),
#                * a note-name legend,
#                * a string / fret reading guide.
# ==========================================================================

"""Render guitar tab notes as ASCII tab strings.

The renderer guarantees three readability properties:

1. **Fixed column width.** Every note column is exactly ``COLUMN_WIDTH`` (3)
   characters, so single-digit and double-digit frets line up perfectly
   across all six strings (``"-3-"`` versus ``"12-"``).
2. **Bar lines every 16 columns.** A ``|`` is inserted after every 16th
   column to mark a measure of 16th notes — the natural rhythmic unit for
   the beginner-friendly tabs this app produces.
3. **Same-time stacking.** If the monophonic path receives multiple notes
   that share the same time (rare with ``monophonic_mode`` on, but
   possible), they are merged into a single column with the fret numbers
   stacked across the relevant string rows — instead of being rendered
   as two adjacent columns, which would falsely suggest a sequence.

Two render entry points share the same body:

    render()         — monophonic: one active string per column.
    render_chords()  — polyphonic: multiple active strings per column.
"""

from __future__ import annotations

from typing import Union

from .fretboard_mapper import TabNote, ChordTabNote, STRING_NAMES

# ---------------------------------------------------------------------------
# Module constants — tuned to the spec.
# ---------------------------------------------------------------------------
# Fixed width of every note column in characters.  3 means a single-digit
# fret renders as "-3-" and a double-digit fret as "12-" — both 3 chars.
COLUMN_WIDTH: int = 3

# How many columns sit inside one measure.  16 corresponds to one measure
# of 16th notes (the finest grid the app currently supports).  Bar lines
# are drawn after every NOTES_PER_BAR columns.
NOTES_PER_BAR: int = 16


# ---------------------------------------------------------------------------
# Note-name helper (no librosa dependency)
# ---------------------------------------------------------------------------
# The twelve chromatic pitch classes in order, used to convert MIDI → name.
_CHROMATIC: list[str] = ["C", "C#", "D", "D#", "E", "F",
                          "F#", "G", "G#", "A", "A#", "B"]


def _format_cell(fret_str: str, width: int = COLUMN_WIDTH) -> str:
    """Format a fret string into a fixed-width column cell.

    The rendered cell is always exactly ``width`` characters wide and
    *always ends in a dash* — this matches the spec's examples:

        "3"  → "-3-"   (one digit, leading dash + trailing dash)
        "12" → "12-"   (two digits, no leading dash, trailing dash)
        "0"  → "-0-"   (open string)

    Achieved by right-justifying the fret string into the first
    ``width - 1`` characters with leading dashes, then appending the
    closing dash separately.  Python's built-in ``str.center`` cannot be
    used because it places the odd-padding character on the *left*,
    producing ``"-12"`` for two-digit frets which would break alignment
    with the trailing-dash convention used by single-digit frets.

    Args:
        fret_str: The fret number as a string (e.g. ``"0"``, ``"12"``).
        width:    Target cell width in characters.  Defaults to
            :data:`COLUMN_WIDTH` (3).

    Returns:
        A ``width``-character cell ending in ``"-"``.
    """
    return fret_str.rjust(width - 1, "-") + "-"


def _midi_to_note_name(midi: int) -> str:
    """Convert a MIDI note number to a human-readable note name.

    Uses the standard formula:
    pitch class = ``midi mod 12``, octave = ``midi // 12 - 1``.

    Args:
        midi: MIDI note number (0–127).

    Returns:
        str: Note name with octave, e.g. 64 → ``'E4'``, 69 → ``'A4'``.
    """
    return f"{_CHROMATIC[midi % 12]}{(midi // 12) - 1}"


# ---------------------------------------------------------------------------
# Internal column descriptors used by the renderer.
# ---------------------------------------------------------------------------
# A *mono* column holds exactly one active string.  Kept as a 3-tuple for
# back-compat with the unit-test API that destructures the tuple.
_MonoCol = tuple[int, str, int]                    # (active_str_idx, fret_str, width)

# A *stacked* column (used both for chords and for collisions in the mono
# path) holds a {string_idx: fret_str} map and a fixed column width.
_StackedCol = tuple[dict[int, str], int]            # (fret_map, width)

# A column produced by the builders may be either flavour.
_AnyCol = Union[_MonoCol, _StackedCol]

# Back-compat alias — older code referred to the chord column as _ChordCol.
_ChordCol = _StackedCol


class TabRenderer:
    """Render tab notes as a human-readable ASCII guitar tab.

    Example::

        r = TabRenderer(notes_per_line=48)        # 3 measures / line
        print(r.render(tab_notes))                # monophonic
        print(r.render_chords(chord_notes))       # polyphonic
    """

    # 48 columns = 3 measures of 16th notes per line.  Matches the
    # spec example which shows three bar-separated chunks on one line.
    DEFAULT_NOTES_PER_LINE: int = 48

    def __init__(self, notes_per_line: int = DEFAULT_NOTES_PER_LINE) -> None:
        """Initialise TabRenderer.

        Args:
            notes_per_line: Note columns per line before wrapping.  For
                cleanest measure alignment this should be a multiple of
                :data:`NOTES_PER_BAR` (16), but any positive value works.
        """
        self.notes_per_line = max(1, int(notes_per_line))

    # ------------------------------------------------------------------
    # Monophonic render
    # ------------------------------------------------------------------

    def render(
        self,
        tab_notes: list[TabNote],
        tempo: float | None = None,
        max_fret: int | None = None,
        one_string_mode: bool = False,
        duration: float | None = None,
        stem: str = "full mix",
    ) -> str:
        """Render monophonic tab notes as ASCII guitar tab.

        Args:
            tab_notes: Output of :meth:`FretboardMapper.map`.
            tempo: Optional BPM for the stats header.
            max_fret: Optional max-fret setting for the settings line.
            one_string_mode: Whether one-string mode was active.
            duration: Optional song duration in seconds.  Falls back to
                the timestamp of the last note when not supplied.
            stem: Source-separation stem name shown in the settings line
                (e.g. ``"vocals"``, ``"other"``).  Defaults to
                ``"full mix"`` when no separation was applied.

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

        # Note count counts every individual played note (not column count,
        # because a stacked column may represent two notes at the same time).
        note_count = len(tab_notes)
        # Approximate duration from the last note's timestamp when the
        # caller did not supply the true audio duration.
        if duration is None:
            duration = max((t for t, *_ in tab_notes), default=0.0)

        return self._render_body(
            columns, note_names, tempo, max_fret, one_string_mode,
            polyphonic=False, note_count=note_count, duration=duration,
            stem=stem,
        )

    def _build_mono_columns(self, tab_notes: list[TabNote]) -> list[_AnyCol]:
        """Convert tab notes into per-column render descriptors.

        Notes that share the **same** ``time`` are merged into a single
        stacked column so they render as one event (one column with
        multiple string rows showing fret numbers) rather than as two
        adjacent columns implying sequence.  With ``monophonic_mode``
        active this collision case is rare, but we handle it correctly
        anyway per the spec.

        Args:
            tab_notes: ``(time, string_idx, fret, midi)`` tuples assumed
                to be in chronological order.

        Returns:
            List of column descriptors — a 3-tuple ``_MonoCol`` for
            single-note columns and a 2-tuple ``_StackedCol`` for
            same-time collisions.
        """
        cols: list[_AnyCol] = []
        i = 0
        n = len(tab_notes)
        while i < n:
            time_t = tab_notes[i][0]
            # Walk forward through every note that shares this exact time.
            j = i + 1
            while j < n and tab_notes[j][0] == time_t:
                j += 1
            group = tab_notes[i:j]
            if len(group) == 1:
                # Common case: emit a 3-tuple mono column.
                _, string_idx, fret, _ = group[0]
                cols.append((string_idx, str(fret), COLUMN_WIDTH))
            else:
                # Collision: stack each note's fret on its respective string.
                # If two notes accidentally land on the same string the
                # later one wins — this is a defensive choice; in practice
                # the FretboardMapper avoids assigning two notes to the
                # same string at the same time.
                fret_map: dict[int, str] = {}
                for _, string_idx, fret, _ in group:
                    fret_map[string_idx] = str(fret)
                cols.append((fret_map, COLUMN_WIDTH))
            i = j
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
        duration: float | None = None,
        stem: str = "full mix",
    ) -> str:
        """Render polyphonic chord notes as ASCII guitar tab.

        Columns with multiple active strings display all fret numbers
        simultaneously, stacked across the relevant string rows — this
        is exactly how a chord is read on a guitar tab.

        Args:
            chord_notes: Output of :meth:`FretboardMapper.map_chords`.
            tempo: Optional BPM for the stats header.
            max_fret: Optional max-fret setting for the settings line.
            one_string_mode: Whether one-string mode was active.
            duration: Optional song duration in seconds.  Falls back to
                the last chord's timestamp when not supplied.
            stem: Source-separation stem name shown in the settings line
                (e.g. ``"vocals"``, ``"other"``).  Defaults to
                ``"full mix"`` when no separation was applied.

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

        # For the note legend: single notes show as "E4", chords as "(E4+G4+B4)".
        note_names: list[str] = []
        for _, _positions, midis in chord_notes:
            if len(midis) == 1:
                note_names.append(_midi_to_note_name(midis[0]))
            else:
                note_names.append(
                    "(" + "+".join(_midi_to_note_name(m) for m in midis) + ")"
                )

        # Total played notes = sum of notes inside every chord event.
        note_count = sum(len(midis) for _, _, midis in chord_notes)
        if duration is None:
            duration = max((t for t, *_ in chord_notes), default=0.0)

        return self._render_body(
            columns, note_names, tempo, max_fret, one_string_mode,
            polyphonic=True, note_count=note_count, duration=duration,
            stem=stem,
        )

    def _build_chord_columns(
        self, chord_notes: list[ChordTabNote]
    ) -> list[_StackedCol]:
        """Build stacked column descriptors for the polyphonic path.

        Every column carries a ``{string_idx: fret_str}`` map and the
        constant :data:`COLUMN_WIDTH`, so column alignment never depends
        on the largest fret in the song.

        Args:
            chord_notes: ``(time, [(string_idx, fret), ...], [midi, ...])``
                tuples.

        Returns:
            List of ``(fret_map, width)`` tuples.
        """
        cols: list[_StackedCol] = []
        for _, positions, _ in chord_notes:
            fret_map = {s: str(f) for s, f in positions}
            cols.append((fret_map, COLUMN_WIDTH))
        return cols

    # ------------------------------------------------------------------
    # Shared rendering core
    # ------------------------------------------------------------------

    def _render_body(
        self,
        columns: list[_AnyCol],
        note_names: list[str],
        tempo: float | None,
        max_fret: int | None,
        one_string_mode: bool,
        polyphonic: bool,
        note_count: int | None = None,
        duration: float | None = None,
        stem: str = "full mix",
    ) -> str:
        """Build the complete ASCII tab string from column descriptors.

        Used by both :meth:`render` and :meth:`render_chords`.  Handles
        the header, line-wrapped tab body with bar lines, the note-name
        legend, and the trailing reading guide.

        Args:
            columns: List of column descriptors (mono or stacked).  The
                renderer dispatches per-column based on the descriptor
                shape, so the two flavours can be intermixed.
            note_names: Human-readable note labels (e.g. ``'E4'`` or
                ``'(E4+G4+B4)'`` for chords).
            tempo: Song tempo in BPM, shown in the stats header.
            max_fret: Max-fret setting, shown in the settings line.
            one_string_mode: Whether one-string mode was active.
            polyphonic: ``True`` when the input came from the chord path —
                only used to add a "Polyphonic (Basic-pitch)" tag to the
                settings line.
            note_count: Total notes for the stats header.  Falls back to
                ``len(columns)`` if omitted.
            duration: Song duration in seconds for the stats header.
                Falls back to ``0.0`` if omitted.

        Returns:
            str: The complete multi-line ASCII guitar tab.
        """
        # --- Line wrapping ---
        # Each chunk becomes one set of 6 string rows.  Bar lines are
        # placed inside a chunk every NOTES_PER_BAR columns by the row
        # builder below.
        chunks = [
            columns[i : i + self.notes_per_line]
            for i in range(0, len(columns), self.notes_per_line)
        ]

        parts: list[str] = []

        # --- Header section ---
        # Decorative banner
        parts.append("=" * 60)
        parts.append("  AutoTabber — Beginner Guitar Tab")
        parts.append("=" * 60)

        # Stats line per spec: "Tempo: 120 BPM  |  Notes: 87  |  Duration: 60.9s"
        stats: list[str] = []
        stats.append(
            f"Tempo: {tempo:.0f} BPM" if tempo is not None else "Tempo: ?"
        )
        stats.append(
            f"Notes: {note_count if note_count is not None else len(columns)}"
        )
        stats.append(f"Duration: {duration if duration is not None else 0.0:.1f}s")
        parts.append("  " + "  |  ".join(stats))

        # Settings line — tuning + any optional flags + provenance
        settings: list[str] = ["Standard tuning (EADGBE)"]
        if max_fret is not None:
            settings.append(f"Max fret: {max_fret}")
        if one_string_mode:
            settings.append("One-string mode")
        if polyphonic:
            settings.append("Polyphonic (Basic-pitch)")
        # ``stem`` is always present (defaults to "full mix") so every
        # rendered tab shows its provenance.  When source separation is
        # used the value is the chosen stem name, e.g. "vocals".
        settings.append(f"Source: {stem}")
        parts.append("  " + " | ".join(settings))
        parts.append("")

        # --- Tab body: 6 string rows per chunk, bar lines every 16 cols ---
        for chunk in chunks:
            for str_idx, name in enumerate(STRING_NAMES):
                # Each row starts with the string name followed by an
                # opening pipe (e.g. ``"e|"``).  Width-1 names like "e"
                # already give consistent two-char prefixes; "B|", "G|"
                # etc. are also two chars.
                row = f"{name}|"
                for col_index, col in enumerate(chunk):
                    # Dispatch by column flavour — a mono column is a
                    # 3-tuple (active_str, fret_str, width); a stacked
                    # / chord column is a 2-tuple (fret_map, width).
                    if isinstance(col[0], dict):
                        fret_map, width = col  # type: ignore[misc]
                        fret_str = fret_map.get(str_idx)
                        if fret_str is not None:
                            # Active string in this stacked column.
                            row += _format_cell(fret_str, width)
                        else:
                            # Inactive string — pure dashes for the column.
                            row += "-" * width
                    else:
                        active_str, fret_str, width = col  # type: ignore[misc]
                        if str_idx == active_str:
                            # _format_cell guarantees a fixed ``width``-char
                            # cell ending in "-": single-digit frets render
                            # as "-3-", double-digit frets as "12-".
                            row += _format_cell(fret_str, width)
                        else:
                            row += "-" * width

                    # Insert a bar line after every NOTES_PER_BAR columns,
                    # but skip the insertion when the bar would fall on
                    # the chunk's closing pipe (avoids "||" at line ends).
                    is_bar_boundary = (col_index + 1) % NOTES_PER_BAR == 0
                    is_chunk_end = (col_index + 1) == len(chunk)
                    if is_bar_boundary and not is_chunk_end:
                        row += "|"

                # Closing pipe terminates every row, regardless of bar
                # alignment — keeps the visual frame consistent.
                row += "|"
                parts.append(row)
            parts.append("")  # blank line between chunks

        # --- Note-name legend ---
        parts.append("Notes:  " + "  ".join(note_names))
        parts.append("")

        # --- Reading-guide footer (per spec) ---
        parts.append("Strings: e=high E (1st), B, G, D, A, E=low E (6th)")
        parts.append("Numbers = fret position. 0 = open string.")
        return "\n".join(parts)
