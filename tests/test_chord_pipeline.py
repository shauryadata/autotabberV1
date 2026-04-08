"""Tests for the polyphonic (chord) pipeline and TabStorage.

No audio files or internet access required.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from beginner_tab.tab_simplifier import TabSimplifier, ChordNote
from beginner_tab.fretboard_mapper import FretboardMapper, ChordTabNote
from beginner_tab.tab_renderer import TabRenderer
from beginner_tab.tab_storage import TabStorage


# ===========================================================================
# TabSimplifier — quantize_chords
# ===========================================================================

class TestQuantizeChords:

    def test_empty_input(self) -> None:
        s = TabSimplifier([])
        assert s.quantize_chords() == []

    def test_single_note_becomes_single_chord(self) -> None:
        notes = [(0.0, 60, 0.9)]
        result = TabSimplifier(notes).quantize_chords()
        assert len(result) == 1
        _, midis, _ = result[0]
        assert midis == [60]

    def test_simultaneous_notes_grouped(self) -> None:
        """Two notes very close in time land in the same slot → one chord."""
        notes = [(0.0, 60, 0.8), (0.02, 64, 0.9)]  # 20 ms apart, same slot
        result = TabSimplifier(notes, tempo=120.0, subdivision=2).quantize_chords()
        assert len(result) == 1
        _, midis, _ = result[0]
        assert set(midis) == {60, 64}

    def test_same_midi_at_same_slot_deduped(self) -> None:
        """Duplicate detections of the same pitch in one slot are deduplicated."""
        notes = [(0.0, 60, 0.7), (0.01, 60, 0.9)]
        result = TabSimplifier(notes, tempo=120.0).quantize_chords()
        assert len(result) == 1
        _, midis, _ = result[0]
        assert midis.count(60) == 1

    def test_notes_at_different_slots_preserved(self) -> None:
        """Notes well apart in time → separate chord events."""
        notes = [(0.0, 60, 0.9), (1.0, 64, 0.9)]
        result = TabSimplifier(notes, tempo=120.0, subdivision=2).quantize_chords()
        assert len(result) == 2

    def test_output_sorted_by_time(self) -> None:
        notes = [(1.0, 67, 0.9), (0.0, 60, 0.8), (0.5, 64, 0.85)]
        result = TabSimplifier(notes, tempo=120.0).quantize_chords()
        times = [n[0] for n in result]
        assert times == sorted(times)

    def test_adjacent_identical_chord_collapsed(self) -> None:
        """Two adjacent slots with the same pitch set collapse into one."""
        # slot_duration at 120 BPM / 8th = 0.25 s; slots 0 and 1
        notes = [(0.0, 60, 0.8), (0.25, 60, 0.9)]
        result = TabSimplifier(notes, tempo=120.0, subdivision=2).quantize_chords()
        assert len(result) == 1

    def test_non_adjacent_same_chord_kept(self) -> None:
        """Same pitches two slots apart are not collapsed (repeated chord)."""
        notes = [(0.0, 60, 0.9), (0.5, 60, 0.9)]  # slots 0 and 2
        result = TabSimplifier(notes, tempo=120.0, subdivision=2).quantize_chords()
        assert len(result) == 2

    def test_midi_list_sorted(self) -> None:
        """MIDI notes within each chord are returned sorted low→high."""
        notes = [(0.0, 67, 0.9), (0.01, 60, 0.8), (0.02, 64, 0.85)]
        result = TabSimplifier(notes, tempo=120.0).quantize_chords()
        _, midis, _ = result[0]
        assert midis == sorted(midis)

    def test_confidence_is_average(self) -> None:
        """Confidence in output is the average of deduplicated notes in slot."""
        notes = [(0.0, 60, 0.8), (0.01, 64, 0.6)]
        result = TabSimplifier(notes, tempo=120.0).quantize_chords()
        _, _, conf = result[0]
        assert 0.6 <= conf <= 0.9


class TestSimplifyRangeChords:

    def test_empty(self) -> None:
        s = TabSimplifier([])
        assert s.simplify_range_chords([]) == []

    def test_in_range_unchanged(self) -> None:
        chords: list[ChordNote] = [(0.0, [60, 64], 0.9)]
        s = TabSimplifier([])
        result = s.simplify_range_chords(chords)
        assert result[0][1] == [60, 64]

    def test_low_note_transposed_up(self) -> None:
        chords: list[ChordNote] = [(0.0, [20, 60], 0.9)]
        s = TabSimplifier([])
        result = s.simplify_range_chords(chords, target_midi_min=40)
        assert all(m >= 40 for m in result[0][1])

    def test_octave_collision_removed(self) -> None:
        """If two notes collapse to the same pitch after transposition, dedup."""
        chords: list[ChordNote] = [(0.0, [40, 52], 0.9)]  # E2 and E3 in range
        s = TabSimplifier([])
        result = s.simplify_range_chords(chords, target_midi_min=40)
        _, midis, _ = result[0]
        assert len(midis) == len(set(midis))


# ===========================================================================
# FretboardMapper — map_chords
# ===========================================================================

class TestMapChords:

    def test_empty_input(self) -> None:
        m = FretboardMapper(max_fret=12)
        assert m.map_chords([]) == []

    def test_single_note_chord(self) -> None:
        chords: list[ChordNote] = [(0.0, [64], 0.9)]  # E4
        m = FretboardMapper(max_fret=12)
        result = m.map_chords(chords)
        assert len(result) == 1
        time, positions, midis = result[0]
        assert len(positions) == 1
        assert midis == [64]

    def test_no_two_notes_on_same_string(self) -> None:
        """A chord must not assign two notes to the same string."""
        chords: list[ChordNote] = [(0.0, [64, 67, 71], 0.9)]  # E4, G4, B4
        m = FretboardMapper(max_fret=12)
        result = m.map_chords(chords)
        _, positions, _ = result[0]
        strings_used = [s for s, _ in positions]
        assert len(strings_used) == len(set(strings_used))

    def test_frets_within_max_fret(self) -> None:
        chords: list[ChordNote] = [(0.0, [64, 67], 0.9)]
        m = FretboardMapper(max_fret=5)
        result = m.map_chords(chords)
        for _, positions, _ in result:
            for _, fret in positions:
                assert fret <= 5

    def test_unmappable_note_skipped(self) -> None:
        """A note with no valid position is dropped; skipped_count updated."""
        # E2 (40) cannot appear on high-e within fret 3
        chords: list[ChordNote] = [(0.0, [40, 64], 0.9)]
        m = FretboardMapper(max_fret=3, one_string_mode=True)
        result = m.map_chords(chords)
        assert m.skipped_count >= 1

    def test_skipped_count_resets(self) -> None:
        m = FretboardMapper(max_fret=3, one_string_mode=True)
        m.map_chords([(0.0, [40], 0.9)])  # skips 1
        assert m.skipped_count == 1
        m.map_chords([(0.0, [64], 0.9)])  # skips 0
        assert m.skipped_count == 0

    def test_output_sorted_by_time(self) -> None:
        chords: list[ChordNote] = [
            (1.0, [64], 0.9), (0.0, [60], 0.8), (0.5, [67], 0.85)
        ]
        m = FretboardMapper(max_fret=12)
        result = m.map_chords(chords)
        times = [t for t, _, _ in result]
        assert times == sorted(times)

    def test_positions_string_order(self) -> None:
        """Within a chord, positions are sorted by string index (low → high)."""
        chords: list[ChordNote] = [(0.0, [64, 59, 55], 0.9)]  # E4, B3, G3
        m = FretboardMapper(max_fret=12)
        result = m.map_chords(chords)
        _, positions, _ = result[0]
        strings = [s for s, _ in positions]
        assert strings == sorted(strings)


# ===========================================================================
# TabRenderer — render_chords
# ===========================================================================

class TestRenderChords:

    def _make_chord_data(self) -> list[ChordTabNote]:
        # E4 (fret 0) on string 0 + B3 (fret 0) on string 1
        return [(0.0, [(0, 0), (1, 0)], [64, 59])]

    def test_empty_returns_message(self) -> None:
        r = TabRenderer()
        out = r.render_chords([])
        assert "No chords" in out

    def test_six_string_rows(self) -> None:
        r = TabRenderer()
        out = r.render_chords(self._make_chord_data())
        string_names = {"e|", "B|", "G|", "D|", "A|", "E|"}
        lines = [l for l in out.split("\n") if l[:2] in string_names]
        assert len(lines) == 6

    def test_both_active_strings_show_frets(self) -> None:
        r = TabRenderer()
        out = r.render_chords(self._make_chord_data())
        lines = [l for l in out.split("\n") if "|" in l]
        e_row = next(l for l in lines if l.startswith("e|"))
        b_row = next(l for l in lines if l.startswith("B|"))
        assert "0" in e_row
        assert "0" in b_row

    def test_inactive_strings_are_dashes(self) -> None:
        data: list[ChordTabNote] = [(0.0, [(0, 0)], [64])]  # only high-e
        r = TabRenderer()
        out = r.render_chords(data)
        lines = [l for l in out.split("\n") if "|" in l]
        e_row = next(l for l in lines if l.startswith("e|"))
        b_row = next(l for l in lines if l.startswith("B|"))
        assert "0" in e_row
        inner = b_row[2:-1]
        assert all(c == "-" for c in inner)

    def test_chord_label_in_notes_footer(self) -> None:
        r = TabRenderer()
        out = r.render_chords(self._make_chord_data())
        assert "Notes:" in out
        assert "(" in out  # chord format uses parentheses

    def test_header_contains_polyphonic_label(self) -> None:
        r = TabRenderer()
        out = r.render_chords(self._make_chord_data())
        assert "Polyphonic" in out or "Basic-pitch" in out

    def test_two_chunks_when_over_limit(self) -> None:
        data: list[ChordTabNote] = [(float(i), [(0, i % 5)], [64 + i]) for i in range(8)]
        r = TabRenderer(notes_per_line=4)
        out = r.render_chords(data)
        e_rows = [l for l in out.split("\n") if l.startswith("e|")]
        assert len(e_rows) == 2


# ===========================================================================
# TabStorage (sqlite3)
# ===========================================================================

class TestTabStorage:

    @pytest.fixture()
    def db(self, tmp_path) -> TabStorage:
        return TabStorage(tmp_path / "test.db")

    def test_empty_list(self, db: TabStorage) -> None:
        assert db.list_tabs() == []

    def test_count_zero(self, db: TabStorage) -> None:
        assert db.count() == 0

    def test_save_returns_id(self, db: TabStorage) -> None:
        tab_id = db.save("song.mp3", "e|---0---|")
        assert isinstance(tab_id, int)
        assert tab_id >= 1

    def test_save_increments_count(self, db: TabStorage) -> None:
        db.save("a.mp3", "tab1")
        db.save("b.mp3", "tab2")
        assert db.count() == 2

    def test_list_tabs_newest_first(self, db: TabStorage) -> None:
        id1 = db.save("first.mp3", "tab1")
        id2 = db.save("second.mp3", "tab2")
        records = db.list_tabs()
        assert records[0]["id"] == id2
        assert records[1]["id"] == id1

    def test_list_tabs_excludes_tab_text(self, db: TabStorage) -> None:
        db.save("song.mp3", "big tab text here")
        records = db.list_tabs()
        assert "tab_text" not in records[0]

    def test_get_tab_includes_text(self, db: TabStorage) -> None:
        tab_id = db.save("song.mp3", "e|---0---|")
        rec = db.get_tab(tab_id)
        assert rec is not None
        assert rec["tab_text"] == "e|---0---|"

    def test_get_tab_not_found_returns_none(self, db: TabStorage) -> None:
        assert db.get_tab(9999) is None

    def test_save_all_metadata(self, db: TabStorage) -> None:
        tab_id = db.save(
            "song.mp3", "tab",
            reference_url="https://example.com",
            detector="basic-pitch",
            tempo=92.0,
            max_fret=7,
            one_string=False,
            note_count=42,
        )
        rec = db.get_tab(tab_id)
        assert rec["reference_url"] == "https://example.com"
        assert rec["detector"] == "basic-pitch"
        assert abs(rec["tempo"] - 92.0) < 1e-6
        assert rec["max_fret"] == 7
        assert rec["one_string"] == 0
        assert rec["note_count"] == 42

    def test_delete_tab(self, db: TabStorage) -> None:
        tab_id = db.save("song.mp3", "tab")
        db.delete_tab(tab_id)
        assert db.get_tab(tab_id) is None
        assert db.count() == 0

    def test_delete_nonexistent_is_silent(self, db: TabStorage) -> None:
        db.delete_tab(9999)  # must not raise

    def test_multiple_saves_independent(self, db: TabStorage) -> None:
        id1 = db.save("a.mp3", "tab_a", tempo=90.0)
        id2 = db.save("b.mp3", "tab_b", tempo=120.0)
        assert db.get_tab(id1)["tab_text"] == "tab_a"
        assert db.get_tab(id2)["tab_text"] == "tab_b"

    def test_created_at_is_string(self, db: TabStorage) -> None:
        tab_id = db.save("song.mp3", "tab")
        rec = db.get_tab(tab_id)
        assert isinstance(rec["created_at"], str)
        assert len(rec["created_at"]) > 0
