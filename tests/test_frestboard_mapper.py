"""Unit tests for FretboardMapper.

These tests exercise position lookup, max-fret enforcement, one-string
mode, and the note-placement greedy algorithm — all without requiring
any audio or librosa.
"""

from __future__ import annotations

import pytest

from beginner_tab.fretboard_mapper import (
    FretboardMapper,
    STANDARD_TUNING_MIDI,
    STRING_NAMES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def mapper_default() -> FretboardMapper:
    return FretboardMapper(max_fret=12)


@pytest.fixture()
def mapper_5() -> FretboardMapper:
    return FretboardMapper(max_fret=5)


@pytest.fixture()
def mapper_3_one_string() -> FretboardMapper:
    return FretboardMapper(max_fret=3, one_string_mode=True)


# ── get_positions ─────────────────────────────────────────────────────────────

class TestGetPositions:

    def test_open_high_e_string(self, mapper_default: FretboardMapper) -> None:
        """E4 (MIDI 64) should appear at fret 0 on the high-e string (index 0)."""
        positions = mapper_default.get_positions(64)
        assert (0, 0) in positions

    def test_open_low_e_string(self, mapper_default: FretboardMapper) -> None:
        """E2 (MIDI 40) should appear at fret 0 on the low-E string (index 5)."""
        positions = mapper_default.get_positions(40)
        assert (5, 0) in positions

    def test_d_on_a_string_fret5(self, mapper_default: FretboardMapper) -> None:
        """D3 (MIDI 50) can be played on the A string (index 4) at fret 5."""
        # A string open = MIDI 45; 50 - 45 = 5
        positions = mapper_default.get_positions(50)
        assert (4, 5) in positions

    def test_d_on_d_string_open(self, mapper_default: FretboardMapper) -> None:
        """D3 (MIDI 50) is also an open D string (index 3, fret 0)."""
        positions = mapper_default.get_positions(50)
        assert (3, 0) in positions

    def test_max_fret_upper_bound(self, mapper_5: FretboardMapper) -> None:
        """No position should have a fret number above max_fret."""
        for midi in range(40, 80):
            for _, fret in mapper_5.get_positions(midi):
                assert fret <= 5

    def test_max_fret_3_excludes_high_frets(self) -> None:
        """A4 (MIDI 69) on high-e would be fret 5 — excluded when max_fret=3."""
        mapper = FretboardMapper(max_fret=3)
        positions = mapper.get_positions(69)
        assert all(fret <= 3 for _, fret in positions)

    def test_max_fret_3_includes_g4(self) -> None:
        """G4 (MIDI 67) on high-e is fret 3 — should be included at max_fret=3."""
        mapper = FretboardMapper(max_fret=3)
        positions = mapper.get_positions(67)
        assert (0, 3) in positions

    def test_note_below_range_returns_empty(
        self, mapper_3_one_string: FretboardMapper
    ) -> None:
        """A very low note (MIDI 20) cannot be played on the high-e string."""
        positions = mapper_3_one_string.get_positions(20)
        assert positions == []

    def test_one_string_mode_only_high_e(
        self, mapper_3_one_string: FretboardMapper
    ) -> None:
        """In one-string mode every returned position uses string index 0."""
        positions = mapper_3_one_string.get_positions(64)
        assert all(s == 0 for s, _ in positions)

    def test_no_negative_frets(self, mapper_default: FretboardMapper) -> None:
        """Fret numbers must never be negative."""
        for midi in range(40, 80):
            for _, fret in mapper_default.get_positions(midi):
                assert fret >= 0

    def test_standard_tuning_constants(self) -> None:
        """Tuning list and string names have the same length."""
        assert len(STANDARD_TUNING_MIDI) == len(STRING_NAMES) == 6


# ── map ───────────────────────────────────────────────────────────────────────

class TestMap:

    def test_empty_input(self, mapper_5: FretboardMapper) -> None:
        assert mapper_5.map([]) == []

    def test_single_note_mapped(self, mapper_5: FretboardMapper) -> None:
        """E4 at fret 0 on high-e should map successfully."""
        result = mapper_5.map([(0.0, 64, 0.9)])
        assert len(result) == 1
        time, string_idx, fret, midi = result[0]
        assert midi == 64
        assert fret == 0
        assert string_idx == 0

    def test_output_tuple_types(self, mapper_5: FretboardMapper) -> None:
        result = mapper_5.map([(0.0, 64, 0.9)])
        time, string_idx, fret, midi = result[0]
        assert isinstance(time, float)
        assert isinstance(string_idx, int)
        assert isinstance(fret, int)
        assert isinstance(midi, int)

    def test_unmappable_note_skipped(
        self, mapper_3_one_string: FretboardMapper
    ) -> None:
        """E2 (MIDI 40) cannot be on high-e within fret 3 — must be skipped."""
        result = mapper_3_one_string.map([(0.0, 40, 0.9)])
        assert result == []

    def test_skipped_count_correct(
        self, mapper_3_one_string: FretboardMapper
    ) -> None:
        notes = [(0.0, 40, 0.9), (0.5, 64, 0.9)]  # 40 is unmappable
        mapper_3_one_string.map(notes)
        assert mapper_3_one_string.skipped_count == 1

    def test_skipped_count_resets_between_calls(self) -> None:
        mapper = FretboardMapper(max_fret=3, one_string_mode=True)
        mapper.map([(0.0, 40, 0.9)])  # 1 skipped
        assert mapper.skipped_count == 1
        mapper.map([(0.0, 64, 0.9)])  # 0 skipped
        assert mapper.skipped_count == 0

    def test_chronological_order_preserved(self, mapper_5: FretboardMapper) -> None:
        notes = [(1.0, 64, 0.9), (0.0, 65, 0.9), (0.5, 67, 0.9)]
        result = mapper_5.map(notes)
        times = [t for t, *_ in result]
        assert times == sorted(times)

    def test_minimal_movement_same_string(self, mapper_5: FretboardMapper) -> None:
        """E4→F4 are adjacent semitones — mapper should keep same string."""
        notes = [(0.0, 64, 0.9), (0.5, 65, 0.9)]
        result = mapper_5.map(notes)
        assert len(result) == 2
        _, s0, _, _ = result[0]
        _, s1, _, _ = result[1]
        # Both notes should land on the same string (high-e, indices 0)
        assert s0 == s1

    def test_max_fret_respected_in_map(self) -> None:
        mapper = FretboardMapper(max_fret=5)
        # A4 (MIDI 69) on high-e = fret 5, which is exactly the limit
        result = mapper.map([(0.0, 69, 0.9)])
        assert len(result) == 1
        _, _, fret, _ = result[0]
        assert fret <= 5

    def test_multiple_notes_all_mapped(self, mapper_5: FretboardMapper) -> None:
        """Simple scale fragment — all notes should be mappable."""
        # E4, F#4, G#4, A4 (pentatonic-ish)
        notes = [(float(i), midi, 0.9) for i, midi in enumerate([64, 66, 68, 69])]
        result = mapper_5.map(notes)
        assert len(result) == 4
