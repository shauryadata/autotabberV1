"""Unit tests for TabRenderer and the _midi_to_note_name helper.

No audio or librosa required — all tests use pre-built TabNote lists.
"""

from __future__ import annotations

import pytest

from beginner_tab.tab_renderer import TabRenderer, _midi_to_note_name
from beginner_tab.fretboard_mapper import STRING_NAMES


# ── _midi_to_note_name ────────────────────────────────────────────────────────

class TestMidiToNoteName:

    def test_middle_c(self) -> None:
        assert _midi_to_note_name(60) == "C4"

    def test_e4_open_high_e(self) -> None:
        assert _midi_to_note_name(64) == "E4"

    def test_e2_open_low_e(self) -> None:
        assert _midi_to_note_name(40) == "E2"

    def test_sharp_note(self) -> None:
        assert _midi_to_note_name(61) == "C#4"

    def test_a4_concert_pitch(self) -> None:
        assert _midi_to_note_name(69) == "A4"

    def test_octave_boundary(self) -> None:
        # B3 (59) → C4 (60) crosses octave boundary
        assert _midi_to_note_name(59) == "B3"
        assert _midi_to_note_name(60) == "C4"

    def test_pitch_class_consistency(self) -> None:
        """Notes 12 semitones apart share the same letter name."""
        name_0 = _midi_to_note_name(48)  # C3
        name_1 = _midi_to_note_name(60)  # C4
        assert name_0[0] == name_1[0] == "C"


# ── TabRenderer ───────────────────────────────────────────────────────────────

class TestTabRendererEmpty:

    def test_empty_input_returns_message(self) -> None:
        r = TabRenderer()
        out = r.render([])
        assert "No notes" in out

    def test_empty_does_not_raise(self) -> None:
        TabRenderer().render([])  # must not raise


class TestTabRendererStructure:
    """Tests for structural properties of the rendered output."""

    # A single note: E4 on high-e string, fret 0, MIDI 64
    SINGLE: list = [(0.0, 0, 0, 64)]

    def _tab_lines(self, tab_notes, **kwargs) -> list[str]:
        """Return only the 6 tab string rows (e|, B|, G|, D|, A|, E|).

        Filtering by string-row prefix (rather than by ``"|" in line``)
        is necessary because the renderer's stats header now legitimately
        contains ``|`` separators between Tempo / Notes / Duration.
        """
        r = TabRenderer(**kwargs)
        output = r.render(tab_notes)
        string_prefixes = tuple(f"{name}|" for name in STRING_NAMES)
        return [ln for ln in output.split("\n") if ln.startswith(string_prefixes)]

    def test_six_string_rows(self) -> None:
        lines = self._tab_lines(self.SINGLE)
        assert len(lines) == 6

    def test_string_name_labels(self) -> None:
        r = TabRenderer()
        output = r.render(self.SINGLE)
        for name in STRING_NAMES:
            assert f"{name}|" in output

    def test_fret_on_correct_string(self) -> None:
        """Fret 3 on string 2 (G) must appear in the G row, not the e row."""
        # string 2 = G string; fret 3; midi 58 (A#3, G string open + 3)
        tab_notes = [(0.0, 2, 3, 58)]
        lines = self._tab_lines(tab_notes)
        g_row = next(ln for ln in lines if ln.startswith("G|"))
        e_row = next(ln for ln in lines if ln.startswith("e|"))
        assert "3" in g_row
        assert "3" not in e_row

    def test_other_strings_are_dashes_only(self) -> None:
        """Strings that are not played should contain only dashes and '|'."""
        tab_notes = [(0.0, 0, 0, 64)]  # only high-e played
        lines = self._tab_lines(tab_notes)
        for ln in lines:
            if not ln.startswith("e|"):
                # Remove string label and pipe chars, remainder should be dashes
                inner = ln[2:-1]  # strip "X|" prefix and trailing "|"
                assert all(c == "-" for c in inner), f"Unexpected chars in: {ln}"

    def test_open_string_renders_zero(self) -> None:
        tab_notes = [(0.0, 0, 0, 64)]
        r = TabRenderer()
        output = r.render(tab_notes)
        e_row = next(ln for ln in output.split("\n") if ln.startswith("e|"))
        assert "0" in e_row

    def test_two_digit_fret_renders_correctly(self) -> None:
        tab_notes = [(0.0, 0, 12, 76)]  # fret 12 on high-e
        r = TabRenderer()
        output = r.render(tab_notes)
        assert "12" in output

    def test_header_present(self) -> None:
        r = TabRenderer()
        output = r.render(self.SINGLE)
        assert "AutoTabber" in output

    def test_note_names_footer_present(self) -> None:
        r = TabRenderer()
        output = r.render(self.SINGLE)
        assert "Notes:" in output

    def test_note_name_in_footer(self) -> None:
        tab_notes = [(0.0, 0, 0, 60)]  # C4
        r = TabRenderer()
        output = r.render(tab_notes)
        assert "C4" in output

    def test_tempo_in_header_when_provided(self) -> None:
        r = TabRenderer()
        output = r.render(self.SINGLE, tempo=98.0)
        assert "98" in output

    def test_max_fret_in_header_when_provided(self) -> None:
        r = TabRenderer()
        output = r.render(self.SINGLE, max_fret=5)
        assert "5" in output


class TestTabRendererLineWrapping:

    def _count_e_rows(self, tab_notes, notes_per_line) -> int:
        r = TabRenderer(notes_per_line=notes_per_line)
        output = r.render(tab_notes)
        return sum(1 for ln in output.split("\n") if ln.startswith("e|"))

    def test_single_chunk_no_wrap(self) -> None:
        notes = [(float(i), 0, i % 6, 60 + i) for i in range(4)]
        assert self._count_e_rows(notes, notes_per_line=8) == 1

    def test_two_chunks_when_over_limit(self) -> None:
        notes = [(float(i), 0, i % 5, 64 + i) for i in range(8)]
        assert self._count_e_rows(notes, notes_per_line=4) == 2

    def test_exact_multiple_produces_correct_chunks(self) -> None:
        notes = [(float(i), 0, i % 5, 64 + i) for i in range(12)]
        assert self._count_e_rows(notes, notes_per_line=4) == 3

    def test_non_multiple_produces_extra_chunk(self) -> None:
        """9 notes with notes_per_line=4 → 3 chunks (4+4+1)."""
        notes = [(float(i), 0, i % 5, 64 + i) for i in range(9)]
        assert self._count_e_rows(notes, notes_per_line=4) == 3


class TestTabRendererColumnWidths:

    def test_single_digit_fret_min_width(self) -> None:
        """Single-digit fret columns should be at least 3 chars wide ('-N-')."""
        r = TabRenderer()
        cols = r._build_mono_columns([(0.0, 0, 5, 69)])
        _, fret_str, width = cols[0]
        assert fret_str == "5"
        assert width >= 3

    def test_two_digit_fret_fixed_column(self) -> None:
        """Per spec, every column is exactly 3 chars — '12' renders as '12-'."""
        r = TabRenderer()
        cols = r._build_mono_columns([(0.0, 0, 12, 76)])
        _, fret_str, width = cols[0]
        assert fret_str == "12"
        assert width == 3  # "12-" — fixed 3-char column for alignment

    def test_zero_fret_width(self) -> None:
        r = TabRenderer()
        cols = r._build_mono_columns([(0.0, 0, 0, 64)])
        _, fret_str, width = cols[0]
        assert fret_str == "0"
        assert width == 3  # '-0-'
