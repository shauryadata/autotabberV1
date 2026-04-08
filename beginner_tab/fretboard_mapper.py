# ==========================================================================
# File:        fretboard_mapper.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Maps MIDI note numbers to physical guitar string/fret
#              positions.  Supports both monophonic (single-note) and
#              polyphonic (chord) mapping.  Uses a beginner-friendly
#              strategy that prefers open strings and lower fret numbers,
#              and respects a configurable max-fret limit.
# ==========================================================================

"""Map MIDI note numbers to guitar fretboard positions.

Two mapping modes share the same tuning / max-fret constraints:

map()          — monophonic: one note → one (string, fret) position.
map_chords()   — polyphonic: a list of simultaneous MIDI notes → multiple
                 (string, fret) positions, one per string (no two notes on
                 the same string at the same time).
"""

from __future__ import annotations

from typing import Optional

from .tab_simplifier import QuantizedNote, ChordNote

# ---------------------------------------------------------------------------
# Tuning constants — standard EADGBE guitar tuning as MIDI note numbers
# ---------------------------------------------------------------------------
# Index 0 = high e string (thinnest, MIDI 64 = E4)
# Index 5 = low E string (thickest, MIDI 40 = E2)
STANDARD_TUNING_MIDI: list[int] = [64, 59, 55, 50, 45, 40]

# Display names for each string, matching the index order above
STRING_NAMES: list[str] = ["e", "B", "G", "D", "A", "E"]

# ---------------------------------------------------------------------------
# Type aliases used throughout the rendering pipeline
# ---------------------------------------------------------------------------
FretPosition = tuple[int, int]                              # (string_idx, fret)
TabNote = tuple[float, int, int, int]                       # (time, string_idx, fret, midi)
ChordTabNote = tuple[float, list[FretPosition], list[int]]  # (time, positions, midis)


class FretboardMapper:
    """Map MIDI pitches to guitar fret positions.

    For monophonic use: call :meth:`map`.
    For polyphonic / chord use: call :meth:`map_chords`.

    Both methods respect ``max_fret`` and ``one_string_mode``.

    Example::

        mapper = FretboardMapper(max_fret=5)
        tab_notes   = mapper.map(quantized_notes)        # monophonic
        chord_notes = mapper.map_chords(chord_notes)     # polyphonic
    """

    def __init__(
        self,
        max_fret: int = 5,
        one_string_mode: bool = False,
        tuning: Optional[list[int]] = None,
    ) -> None:
        """Initialise FretboardMapper.

        Args:
            max_fret: Maximum fret number allowed (e.g. 3, 5, 7, 12).
            one_string_mode: Only use the high-e string (index 0).
            tuning: Open-string MIDI notes high → low.  Defaults to
                standard EADGBE tuning.
        """
        # Highest fret number any note is allowed to use
        self.max_fret = int(max_fret)
        # If True, only the high-e string (index 0) is considered
        self.one_string_mode = one_string_mode
        # Open-string MIDI notes ordered high → low (default: standard EADGBE)
        self.tuning: list[int] = tuning if tuning is not None else STANDARD_TUNING_MIDI
        # Indices of strings available for placement
        self._active_strings: list[int] = (
            [0] if one_string_mode else list(range(len(self.tuning)))
        )
        # Counter: how many notes were dropped in the last map/map_chords call
        self._last_skipped: int = 0

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    # Open-string MIDI pitches for quick lookup (standard EADGBE)
    _OPEN_STRING_MIDIS: frozenset[int] = frozenset(STANDARD_TUNING_MIDI)

    def get_positions(self, midi_note: int) -> list[FretPosition]:
        """Return all valid ``(string_idx, fret)`` pairs for *midi_note*.

        Positions are sorted beginner-friendly: open strings first, then
        by ascending fret number so the easiest fingering comes first.

        Args:
            midi_note: MIDI note number 0–127.

        Returns:
            List of ``(string_index, fret)`` within ``[0, max_fret]``,
            sorted with open strings first, then by fret ascending.
        """
        positions: list[FretPosition] = []
        for s in self._active_strings:
            # fret = target MIDI minus the string's open-string MIDI
            # e.g. MIDI 64 on string 0 (open E4 = 64) → fret 0
            # e.g. MIDI 67 on string 0 (open E4 = 64) → fret 3
            fret = midi_note - self.tuning[s]
            if 0 <= fret <= self.max_fret:
                positions.append((s, fret))
        # Sort beginner-friendly: open strings (fret 0) first, then ascending fret
        positions.sort(key=lambda p: (p[1] != 0, p[1], p[0]))
        return positions

    def get_all_positions(self, midi_note: int) -> list[FretPosition]:
        """Return every valid string/fret combination for *midi_note*.

        Unlike :meth:`get_positions`, this method ignores
        ``one_string_mode`` and always checks all six strings.  Useful
        for displaying alternate fingerings to the user.

        Args:
            midi_note: MIDI note number 0–127.

        Returns:
            List of ``(string_index, fret)`` within ``[0, max_fret]``,
            sorted by fret number ascending, then string index ascending.
        """
        positions: list[FretPosition] = []
        for s in range(len(self.tuning)):
            fret = midi_note - self.tuning[s]
            if 0 <= fret <= self.max_fret:
                positions.append((s, fret))
        positions.sort(key=lambda p: (p[1], p[0]))
        return positions

    # ------------------------------------------------------------------
    # Monophonic mapping
    # ------------------------------------------------------------------

    def map(self, notes: list[QuantizedNote]) -> list[TabNote]:
        """Map quantised monophonic notes to fretboard positions.

        Minimises hand movement using cost ``|Δstring|×3 + |Δfret|``.
        Input is sorted by time defensively.

        Args:
            notes: Output of :meth:`TabSimplifier.quantize`.

        Returns:
            List of ``(time, string_idx, fret, midi)`` tuples.
        """
        self._last_skipped = 0
        result: list[TabNote] = []
        prev_string: Optional[int] = None
        prev_fret: Optional[int] = None

        for time, midi, _conf in sorted(notes, key=lambda n: n[0]):
            positions = self.get_positions(midi)
            if not positions:
                self._last_skipped += 1
                continue
            string_idx, fret = self._best_mono_position(positions, prev_string, prev_fret)
            result.append((time, string_idx, fret, midi))
            prev_string, prev_fret = string_idx, fret

        return result

    def _best_mono_position(
        self,
        positions: list[FretPosition],
        prev_string: Optional[int],
        prev_fret: Optional[int],
    ) -> FretPosition:
        """Pick the most beginner-friendly position from *positions*.

        Ranking criteria (in priority order):

        1. Open strings (fret 0) get a strong bonus.
        2. Lower fret numbers are preferred over higher ones.
        3. When there is a previous note, hand-movement cost is used
           as a tiebreaker: ``|delta_string| * 3 + |delta_fret|``.

        Args:
            positions: Non-empty list of candidate ``(string_idx, fret)``.
            prev_string: String index of the previous note, or ``None``.
            prev_fret: Fret of the previous note, or ``None``.

        Returns:
            The single best ``(string_idx, fret)`` for a beginner.
        """
        if prev_string is None or prev_fret is None:
            # No history — pick open string first, then lowest fret
            return min(positions, key=lambda p: (p[1] != 0, p[1], p[0]))

        def cost(p: FretPosition) -> tuple[int, float, int]:
            is_open = 0 if p[1] == 0 else 1        # open string bonus
            movement = abs(p[0] - prev_string) * 3.0 + abs(p[1] - prev_fret)
            return (is_open, movement, p[1])

        return min(positions, key=cost)

    @property
    def skipped_count(self) -> int:
        """Notes skipped in the last :meth:`map` or :meth:`map_chords` call."""
        return self._last_skipped

    # ------------------------------------------------------------------
    # Polyphonic / chord mapping
    # ------------------------------------------------------------------

    def map_chords(self, chords: list[ChordNote]) -> list[ChordTabNote]:
        """Map polyphonic chord notes to fretboard positions.

        Each chord is assigned to a set of strings with no two notes on
        the same string.  Notes are processed high→low; each is placed on
        the lowest-index (highest-pitched) available string so the chord
        voicing spreads naturally from treble to bass strings.

        Notes that cannot be placed are silently dropped; the count is
        accumulated in :attr:`skipped_count`.

        Args:
            chords: Output of :meth:`TabSimplifier.quantize_chords`.

        Returns:
            List of ``(time, [(string_idx, fret), …], [midi, …])`` tuples.
        """
        self._last_skipped = 0
        result: list[ChordTabNote] = []

        for time, midi_list, _conf in sorted(chords, key=lambda c: c[0]):
            positions, placed_midis = self._assign_chord(midi_list)
            self._last_skipped += len(midi_list) - len(placed_midis)
            if positions:
                result.append((time, positions, placed_midis))

        return result

    def _assign_chord(
        self, midi_notes: list[int]
    ) -> tuple[list[FretPosition], list[int]]:
        """Greedily assign strings to chord notes (no string shared).

        Process notes from highest pitch to lowest.  For each note pick
        the available position that is most beginner-friendly: open
        strings first, then lowest fret, then lowest string index.

        Args:
            midi_notes: List of MIDI note numbers forming the chord.

        Returns:
            ``(positions, placed_midis)`` — parallel lists sorted by
            string index so the tab renders top-string first.
        """
        used: set[int] = set()
        pairs: list[tuple[FretPosition, int]] = []

        for midi in sorted(midi_notes, reverse=True):   # high → low
            candidates = [
                (s, f) for s, f in self.get_positions(midi) if s not in used
            ]
            if not candidates:
                continue
            # Open string first, then lowest fret, then lowest string index
            best = min(candidates, key=lambda p: (p[1] != 0, p[1], p[0]))
            pairs.append((best, midi))
            used.add(best[0])

        # Sort by string index so tab renders top-string first
        pairs.sort(key=lambda x: x[0][0])
        if pairs:
            positions_out, midis_out = zip(*pairs)
            return list(positions_out), list(midis_out)
        return [], []
