"""Unit tests for TabSimplifier.

All tests are pure Python — no audio or librosa required.
"""

from __future__ import annotations

import pytest

from beginner_tab.tab_simplifier import TabSimplifier


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_simplifier(notes, tempo=120.0, subdivision=2) -> TabSimplifier:
    return TabSimplifier(notes, tempo=tempo, subdivision=subdivision)


# ── slot_duration property ────────────────────────────────────────────────────

class TestSlotDuration:

    def test_120bpm_subdivision2(self) -> None:
        s = make_simplifier([], tempo=120.0, subdivision=2)
        # 60 / (120 * 2) = 0.25 s
        assert abs(s.slot_duration - 0.25) < 1e-9

    def test_60bpm_subdivision1(self) -> None:
        s = make_simplifier([], tempo=60.0, subdivision=1)
        # 60 / (60 * 1) = 1.0 s
        assert abs(s.slot_duration - 1.0) < 1e-9

    def test_tempo_below_one_clamped(self) -> None:
        s = make_simplifier([], tempo=0.0)
        # tempo is clamped to 1.0 — should not divide by zero
        assert s.slot_duration > 0


# ── quantize ──────────────────────────────────────────────────────────────────

class TestQuantize:

    def test_empty_returns_empty(self) -> None:
        s = make_simplifier([])
        assert s.quantize() == []

    def test_single_note_passes_through(self) -> None:
        notes = [(0.5, 60, 0.9)]
        result = make_simplifier(notes).quantize()
        assert len(result) == 1
        assert result[0][1] == 60

    def test_output_sorted_by_time(self) -> None:
        notes = [(1.0, 64, 0.9), (0.0, 60, 0.8), (0.5, 62, 0.85)]
        result = make_simplifier(notes).quantize()
        times = [n[0] for n in result]
        assert times == sorted(times)

    def test_rapid_notes_collapse_to_one_slot(self) -> None:
        """Two notes 50 ms apart at 120 BPM/8th-note grid land in the same slot."""
        notes = [(0.0, 60, 0.8), (0.05, 60, 0.9)]
        result = make_simplifier(notes).quantize()
        # Both fall in slot 0 (slot_duration=0.25 s)
        assert len(result) == 1
        assert result[0][1] == 60

    def test_highest_confidence_wins_same_slot(self) -> None:
        """When two different pitches land in the same slot, the higher-confidence
        one is kept."""
        notes = [(0.0, 60, 0.7), (0.05, 62, 0.95)]
        result = make_simplifier(notes).quantize()
        assert len(result) == 1
        assert result[0][1] == 62  # 62 had higher confidence

    def test_consecutive_same_pitch_collapsed(self) -> None:
        """Adjacent slots with the same MIDI pitch collapse to a single event."""
        # slot 0 and slot 1 both carry MIDI 60
        notes = [(0.0, 60, 0.8), (0.25, 60, 0.9)]
        result = make_simplifier(notes, tempo=120.0, subdivision=2).quantize()
        assert len(result) == 1
        assert result[0][1] == 60

    def test_different_pitches_not_collapsed(self) -> None:
        notes = [(0.0, 60, 0.9), (0.25, 62, 0.9)]
        result = make_simplifier(notes, tempo=120.0, subdivision=2).quantize()
        assert len(result) == 2

    def test_collapse_keeps_earlier_time(self) -> None:
        """The collapsed note retains the time of the first occurrence."""
        notes = [(0.0, 60, 0.8), (0.25, 60, 0.9)]
        result = make_simplifier(notes, tempo=120.0, subdivision=2).quantize()
        assert result[0][0] == pytest.approx(0.0)

    def test_three_notes_two_collapsed(self) -> None:
        """[60, 60, 62] → [60, 62] after collapse."""
        notes = [(0.0, 60, 0.9), (0.25, 60, 0.8), (0.5, 62, 0.9)]
        result = make_simplifier(notes, tempo=120.0, subdivision=2).quantize()
        assert len(result) == 2
        assert result[0][1] == 60
        assert result[1][1] == 62

    def test_midi_preserved(self) -> None:
        notes = [(0.0, 69, 0.85)]
        result = make_simplifier(notes).quantize()
        assert result[0][1] == 69

    def test_confidence_preserved_from_best(self) -> None:
        notes = [(0.0, 60, 0.7), (0.05, 60, 0.95)]
        result = make_simplifier(notes).quantize()
        assert result[0][2] == pytest.approx(0.95)


# ── simplify_range ────────────────────────────────────────────────────────────

class TestSimplifyRange:

    def test_empty_returns_empty(self) -> None:
        s = make_simplifier([])
        assert s.simplify_range([]) == []

    def test_note_in_range_unchanged(self) -> None:
        notes = [(0.0, 60, 0.9)]
        s = make_simplifier(notes)
        result = s.simplify_range(notes, target_midi_min=40, target_midi_max=88)
        assert result[0][1] == 60

    def test_low_note_transposed_up(self) -> None:
        notes = [(0.0, 20, 0.9)]   # way below E2 (40)
        s = make_simplifier(notes)
        result = s.simplify_range(notes, target_midi_min=40, target_midi_max=88)
        assert result[0][1] >= 40

    def test_high_note_transposed_down(self) -> None:
        notes = [(0.0, 95, 0.9)]   # above 88
        s = make_simplifier(notes)
        result = s.simplify_range(notes, target_midi_min=40, target_midi_max=88)
        assert result[0][1] <= 88

    def test_transposition_preserves_pitch_class(self) -> None:
        """Octave transposition must not change the note name (pitch class)."""
        notes = [(0.0, 28, 0.9)]   # E1 — 2 octaves below E3
        s = make_simplifier(notes)
        result = s.simplify_range(notes, target_midi_min=40, target_midi_max=88)
        assert result[0][1] % 12 == 28 % 12  # same pitch class

    def test_time_and_confidence_preserved(self) -> None:
        notes = [(1.23, 20, 0.77)]
        s = make_simplifier(notes)
        result = s.simplify_range(notes, target_midi_min=40, target_midi_max=88)
        assert result[0][0] == pytest.approx(1.23)
        assert result[0][2] == pytest.approx(0.77)

    def test_multiple_notes_all_in_range(self) -> None:
        notes = [(float(i), 60, 0.9) for i in range(5)]
        s = make_simplifier(notes)
        result = s.simplify_range(notes, target_midi_min=40, target_midi_max=88)
        assert all(40 <= n[1] <= 88 for n in result)
