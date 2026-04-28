# ==========================================================================
# File:        fretboard_mapper.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Maps MIDI note numbers to physical guitar string/fret
#              positions using a beginner-friendly weighted scoring system.
#              For every MIDI note we generate every reachable
#              (string, fret) combination on the fretboard, score each
#              candidate against a priority-ordered set of rules (open
#              strings preferred, lower frets preferred, large hand jumps
#              penalised, etc.), and pick the highest-scoring position.
#              The scorer also tracks the previous note's position so it
#              can punish unnecessary leaps and reward staying on the
#              same string.
# ==========================================================================

"""Score-based mapping from MIDI pitch to guitar fretboard positions.

The :class:`FretboardMapper` produces beginner-friendly tabs by ranking
every reachable ``(string, fret)`` candidate for a note against a set of
weighted rules:

==========================================  ============================
Rule                                          Score adjustment
==========================================  ============================
Open string (``fret == 0``)                   +100
Low-fret bonus                                +(20 - fret)   (linear)
Fret above ``max_fret_limit``                 -1000          (excludes)
Jump > 5 frets from previous note             -30
Same string as previous note                  +10
==========================================  ============================

The picker selects the highest-scoring candidate.  Scores are floats so
new rules can be added later without colliding with these integer
weights.

Public API:
    * :class:`FretboardMapper`  — score-based mapper for mono and chords.
    * :func:`FretboardMapper.get_all_positions` — every reachable position
      for a MIDI note, sorted best-first by score.
    * :func:`FretboardMapper.get_positions` — playable positions only
      (within ``max_fret_limit`` and the active strings), sorted best-first.
    * :func:`FretboardMapper.score_position` — exposed so unit tests and
      advanced callers can inspect why one position beats another.
"""

from __future__ import annotations

from typing import Optional

from .tab_simplifier import QuantizedNote, ChordNote

# ---------------------------------------------------------------------------
# Tuning constants — standard EADGBE guitar tuning as MIDI note numbers
# ---------------------------------------------------------------------------
# Index 0 = high e string (thinnest, MIDI 64 = E4).
# Index 5 = low E string  (thickest, MIDI 40 = E2).
STANDARD_TUNING_MIDI: list[int] = [64, 59, 55, 50, 45, 40]

# Display names for each string, matching the index order above.
STRING_NAMES: list[str] = ["e", "B", "G", "D", "A", "E"]

# Highest physical fret considered when generating candidate positions.
# Real guitars top out around fret 22-24; we cap candidate generation at
# 24 so the scorer can still see "out of beginner range" hits and dock
# them with the -1000 penalty (rather than them silently disappearing).
PHYSICAL_FRET_CAP: int = 24

# ---------------------------------------------------------------------------
# Scoring weights — centralised so rules can be tweaked or unit-tested
# without hunting through the algorithm.  Names mirror the spec's wording.
# ---------------------------------------------------------------------------
SCORE_OPEN_STRING: float = 100.0    # fret == 0
SCORE_LOW_FRET_BASE: float = 20.0   # adds (20 - fret); lower fret = bigger bonus
SCORE_OVER_LIMIT: float = -1000.0   # fret > max_fret_limit (effective exclusion)
SCORE_BIG_JUMP: float = -30.0       # |fret - prev_fret| > JUMP_THRESHOLD
SCORE_SAME_STRING: float = 10.0     # string_idx == prev_string
JUMP_THRESHOLD: int = 5             # frets — anything strictly greater is "big"

# ---------------------------------------------------------------------------
# Type aliases used throughout the rendering pipeline
# ---------------------------------------------------------------------------
FretPosition = tuple[int, int]                              # (string_idx, fret)
TabNote = tuple[float, int, int, int]                       # (time, string_idx, fret, midi)
ChordTabNote = tuple[float, list[FretPosition], list[int]]  # (time, positions, midis)


class FretboardMapper:
    """Score-based MIDI-to-fret mapper.

    For monophonic use call :meth:`map`; for polyphonic chords call
    :meth:`map_chords`.  Both honour ``max_fret_limit`` and
    ``one_string_mode``.

    Example::

        mapper      = FretboardMapper(max_fret_limit=7)
        tab_notes   = mapper.map(quantized_notes)        # monophonic
        chord_notes = mapper.map_chords(chord_notes)     # polyphonic
    """

    def __init__(
        self,
        max_fret_limit: int = 7,
        one_string_mode: bool = False,
        tuning: Optional[list[int]] = None,
        max_fret: Optional[int] = None,
    ) -> None:
        """Initialise FretboardMapper.

        Args:
            max_fret_limit: Highest fret allowed in the final tab.  Notes
                that can only be played above this fret receive the
                ``SCORE_OVER_LIMIT`` (-1000) penalty in the scorer and
                are filtered out of the playable position list.  Default
                ``7`` (Intermediate).  Use ``5`` for Beginner mode and
                ``12`` for Advanced.
            one_string_mode: When ``True``, restrict placements to the
                high-e string (index 0).  Useful for absolute beginners.
            tuning: Open-string MIDI notes ordered high → low.  Defaults
                to standard EADGBE.
            max_fret: **Backwards-compat alias** for ``max_fret_limit``.
                If supplied, it overrides ``max_fret_limit`` so existing
                callers and unit tests using ``max_fret=...`` keep
                working without modification.
        """
        # Honour the legacy keyword if the caller used it.
        if max_fret is not None:
            max_fret_limit = max_fret

        self.max_fret_limit: int = int(max_fret_limit)
        # Public alias kept so legacy code reading ``mapper.max_fret``
        # still sees the same value.
        self.max_fret: int = self.max_fret_limit

        self.one_string_mode: bool = bool(one_string_mode)
        self.tuning: list[int] = (
            tuning if tuning is not None else STANDARD_TUNING_MIDI
        )

        # Indices of strings the mapper is allowed to place notes on.
        self._active_strings: list[int] = (
            [0] if self.one_string_mode else list(range(len(self.tuning)))
        )

        # Counter of notes dropped during the most recent map / map_chords.
        self._last_skipped: int = 0

        # Previous note's position — used by the scorer to penalise large
        # leaps (rule #4) and reward staying on the same string (rule #5).
        # Both are reset to ``None`` at the start of every :meth:`map`
        # call ("new song"), per the spec's rule #6.
        self._prev_string: Optional[int] = None
        self._prev_fret: Optional[int] = None

    # ------------------------------------------------------------------
    # Position generation & scoring
    # ------------------------------------------------------------------

    def _candidate_positions(
        self, midi_note: int, *, all_strings: bool = False
    ) -> list[FretPosition]:
        """Return every physically reachable ``(string, fret)`` for *midi_note*.

        "Physically reachable" means ``0 <= fret <= PHYSICAL_FRET_CAP`` —
        unlike :meth:`get_positions` this method does **not** apply the
        ``max_fret_limit`` filter.  The scorer expresses that limit as
        the -1000 penalty (per spec rule #2) rather than a hard cut-off,
        so :meth:`get_all_positions` can still surface high-fret
        alternatives if a UI wants to display them.

        Args:
            midi_note: Target MIDI pitch (0-127).
            all_strings: If ``True``, ignore ``one_string_mode`` and
                consider every string of the tuning.

        Returns:
            Unsorted list of ``(string_idx, fret)`` tuples.
        """
        strings = (
            list(range(len(self.tuning))) if all_strings else self._active_strings
        )
        positions: list[FretPosition] = []
        for s in strings:
            # fret = target MIDI minus the open-string MIDI.  Negative
            # values (note below the open string) are physically
            # impossible and silently dropped.
            fret = midi_note - self.tuning[s]
            if 0 <= fret <= PHYSICAL_FRET_CAP:
                positions.append((s, fret))
        return positions

    def score_position(
        self,
        position: FretPosition,
        prev_string: Optional[int] = None,
        prev_fret: Optional[int] = None,
    ) -> float:
        """Return the beginner-friendliness score of a single ``(string, fret)``.

        Higher is better.  Rules are applied in spec order:

            1. **+100**       if ``fret == 0`` — open strings need no
               fingering and are the easiest position for a beginner.
            2. **+(20-fret)** linear bonus — lower frets are physically
               closer to the head of the neck, so easier to find.  Fret
               20 yields 0; anything below cancels out the bonus.
            3. **-1000**      if ``fret > max_fret_limit`` — effectively
               removes the position from contention because no
               combination of other bonuses can recover it.
            4. **-30**        if ``abs(fret - prev_fret) > 5`` — punishes
               big hand jumps that are awkward for beginners.  Skipped
               when there is no previous note (start of song).
            5. **+10**        if ``string == prev_string`` — staying on
               the same string keeps tabs visually clean and the picking
               hand stationary.  Skipped when there is no previous note.

        Args:
            position: The ``(string_idx, fret)`` pair under consideration.
            prev_string: String index of the previously placed note, or
                ``None`` when this is the first note of the piece.
            prev_fret:   Fret of the previously placed note, or ``None``.

        Returns:
            A floating-point score.  The mapper picks the candidate with
            the highest score for each note.
        """
        string_idx, fret = position
        score: float = 0.0

        # Rule 1 — open-string bonus (only when literally fret 0).
        if fret == 0:
            score += SCORE_OPEN_STRING

        # Rule 2 — linear low-fret bonus.  Note: this is in *addition*
        # to the open-string bonus, so a fret-0 position scores 120, a
        # fret-3 position scores 17, and a fret-20 position scores 0.
        score += (SCORE_LOW_FRET_BASE - fret)

        # Rule 3 — over-limit penalty.  The magnitude is large enough
        # that no legal in-range candidate can ever lose to one of these.
        if fret > self.max_fret_limit:
            score += SCORE_OVER_LIMIT

        # Rules 4 & 5 — only meaningful when we actually have a previous
        # placement to compare against.  Skipped at the start of a song.
        if prev_string is not None and prev_fret is not None:
            if abs(fret - prev_fret) > JUMP_THRESHOLD:
                score += SCORE_BIG_JUMP
            if string_idx == prev_string:
                score += SCORE_SAME_STRING

        return score

    def get_positions(self, midi_note: int) -> list[FretPosition]:
        """Return playable positions for *midi_note*, sorted best-first.

        "Playable" here means within ``[0, max_fret_limit]`` and on one
        of the currently active strings (so :attr:`one_string_mode` is
        honoured).  The list is sorted by descending context-free score
        so the most beginner-friendly fingering is at index 0.

        Args:
            midi_note: MIDI note number 0–127.

        Returns:
            List of ``(string_index, fret)`` within the user's max-fret
            limit, sorted best-first by :meth:`score_position` (with no
            previous-note context).  May be empty if the note cannot be
            played within the current constraints.
        """
        positions = [
            p for p in self._candidate_positions(midi_note)
            if p[1] <= self.max_fret_limit
        ]
        # Sort key: primary = -score (higher score first); secondary
        # tiebreakers favour lower fret, then lower string index.
        positions.sort(key=lambda p: (-self.score_position(p), p[1], p[0]))
        return positions

    def get_all_positions(self, midi_note: int) -> list[FretPosition]:
        """Return every reachable position for *midi_note*, sorted best-first.

        Always considers all six strings (ignores ``one_string_mode``)
        so callers can offer the user alternate fingerings.  Positions
        above ``max_fret_limit`` are still included in the output —
        they're only soft-excluded by the -1000 score, so they sort to
        the bottom.

        Args:
            midi_note: MIDI note number 0–127.

        Returns:
            Every reachable ``(string_index, fret)`` for the note,
            sorted by descending :meth:`score_position`.  Out-of-limit
            positions appear last.
        """
        positions = self._candidate_positions(midi_note, all_strings=True)
        positions.sort(key=lambda p: (-self.score_position(p), p[1], p[0]))
        return positions

    # ------------------------------------------------------------------
    # Previous-note bookkeeping
    # ------------------------------------------------------------------

    def reset_position(self) -> None:
        """Forget the last placed note.

        Call this at the start of a new song so the scorer does not
        penalise the very first note for "leaping" away from whatever
        the mapper happened to be playing on the previous file.
        :meth:`map` already invokes this internally; the public method
        is exposed so callers can also clear state mid-stream (for
        example between sections of the same audio file).
        """
        self._prev_string = None
        self._prev_fret = None

    # ------------------------------------------------------------------
    # Monophonic mapping
    # ------------------------------------------------------------------

    def map(self, notes: list[QuantizedNote]) -> list[TabNote]:
        """Map quantised monophonic notes to fretboard positions.

        For every note all candidate positions are generated and scored;
        the highest scorer wins.  Previous-note position is tracked so
        the scorer can apply the big-jump penalty and the same-string
        bonus.  The previous-note state is **reset** at the start of
        every call (one call = one new song).

        Notes that have no playable position (nothing within
        ``max_fret_limit`` on an active string) are dropped; the count
        is exposed via :attr:`skipped_count`.

        Args:
            notes: Output of :meth:`TabSimplifier.quantize`.

        Returns:
            List of ``(time, string_idx, fret, midi)`` tuples in
            chronological order.
        """
        self._last_skipped = 0

        # Spec rule #6: "Reset this at the start of each new song."
        self.reset_position()

        result: list[TabNote] = []
        # Defensive sort — quantize() should already produce sorted
        # output, but a misbehaving caller shouldn't break the scorer.
        for time, midi, _conf in sorted(notes, key=lambda n: n[0]):
            positions = self.get_positions(midi)
            if not positions:
                self._last_skipped += 1
                continue
            string_idx, fret = self._best_mono_position(positions)
            result.append((time, string_idx, fret, midi))
            # Remember this position so the next note's scoring can react.
            self._prev_string = string_idx
            self._prev_fret = fret

        return result

    def _best_mono_position(
        self, positions: list[FretPosition]
    ) -> FretPosition:
        """Pick the highest-scoring candidate using the current prev state.

        The scoring function is :meth:`score_position`.  Ties are broken
        by lower fret, then lower string index, so behaviour is fully
        deterministic.

        Args:
            positions: Non-empty list of candidate ``(string_idx, fret)``.
                Should already be filtered to within ``max_fret_limit``
                (i.e. the output of :meth:`get_positions`).

        Returns:
            The single best ``(string_idx, fret)`` for the next placement.
        """
        # Build a sort key:  highest score wins, then lower fret, then
        # lower string index.  ``max`` prefers the *largest* tuple, so
        # the tiebreakers are negated to re-establish "smaller is better".
        return max(
            positions,
            key=lambda p: (
                self.score_position(p, self._prev_string, self._prev_fret),
                -p[1],   # prefer lower fret
                -p[0],   # prefer lower string index (closer to high-e)
            ),
        )

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
        the same string.  Notes are processed high → low; each is placed
        on the lowest-index (highest-pitched) available string so the
        chord voicing spreads naturally from treble to bass strings.
        Notes that cannot be placed are silently dropped; their count
        accumulates in :attr:`skipped_count`.

        The polyphonic path uses a simpler "open / lowest-fret / lowest-
        string" tiebreak rather than the full prev-aware scorer — a
        chord is a vertical event so the previous-note continuity rules
        are not meaningful.

        Args:
            chords: Output of :meth:`TabSimplifier.quantize_chords`.

        Returns:
            List of ``(time, [(string_idx, fret), ...], [midi, ...])``
            tuples, time-sorted.
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
        """Greedily assign strings to chord notes (no string is shared).

        Process notes from highest pitch to lowest.  For each note pick
        the available position that is most beginner-friendly: open
        strings first, then lowest fret, then lowest string index.

        Args:
            midi_notes: List of MIDI note numbers forming the chord.

        Returns:
            ``(positions, placed_midis)`` — parallel lists sorted by
            string index so the renderer prints top-string first.
        """
        used: set[int] = set()
        pairs: list[tuple[FretPosition, int]] = []

        for midi in sorted(midi_notes, reverse=True):  # high → low pitch
            candidates = [
                (s, f) for s, f in self.get_positions(midi) if s not in used
            ]
            if not candidates:
                continue
            # Open string first, then lowest fret, then lowest string index.
            best = min(candidates, key=lambda p: (p[1] != 0, p[1], p[0]))
            pairs.append((best, midi))
            used.add(best[0])

        # Sort by string index so tab renders top-string first.
        pairs.sort(key=lambda x: x[0][0])
        if pairs:
            positions_out, midis_out = zip(*pairs)
            return list(positions_out), list(midis_out)
        return [], []
