"""Tests for the optional Demucs source-separation stage.

The first test (module-load) runs in every environment — including
Streamlit Cloud where ``demucs`` is intentionally not installed — and
just verifies that ``SOURCE_SEPARATOR_AVAILABLE`` is a real boolean.
The remaining three tests skip cleanly when demucs is absent.

The "invalid stem" test deliberately does NOT load any model, since
``SourceSeparator.separate`` validates the requested stem name *before*
running the lazy model load.  That keeps the test fast even on systems
where demucs is installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

# This import always succeeds — ``SOURCE_SEPARATOR_AVAILABLE`` is set by
# beginner_tab/__init__.py at package import time, so it exists whether
# or not demucs is installed.
from beginner_tab import SOURCE_SEPARATOR_AVAILABLE


# Reusable skip marker so the three demucs-dependent tests share one
# decorator and one explanatory message.
_skip_no_demucs = pytest.mark.skipif(
    not SOURCE_SEPARATOR_AVAILABLE,
    reason=(
        "Demucs is not installed in this environment "
        "(SOURCE_SEPARATOR_AVAILABLE is False).  Install with "
        "`pip install -r requirements-local.txt` to run these tests."
    ),
)


# ===========================================================================
# Always-on test — verifies the conditional-export plumbing in __init__.py
# ===========================================================================

class TestSourceSeparatorModuleLoads:

    def test_source_separator_module_loads_cleanly(self) -> None:
        """``SOURCE_SEPARATOR_AVAILABLE`` is always a real bool.

        This is the contract that the rest of the codebase relies on:
        any caller can check it without first importing the optional
        symbols, and the value is True/False — never None or anything
        truthy-but-not-bool.  Runs in every environment.
        """
        assert isinstance(SOURCE_SEPARATOR_AVAILABLE, bool), (
            f"expected bool, got {type(SOURCE_SEPARATOR_AVAILABLE).__name__}"
        )


# ===========================================================================
# Demucs-dependent tests — skip when the optional dep is missing
# ===========================================================================

@_skip_no_demucs
class TestSourceSeparatorWithDemucs:
    """Behavioural tests that need the real demucs library installed."""

    # ------------------------------------------------------------------
    # get_available_stems — no model load required
    # ------------------------------------------------------------------

    def test_get_available_stems_returns_four_stems(self) -> None:
        """Stem list is the spec-mandated four, in the spec-mandated order."""
        from beginner_tab import SourceSeparator

        sep = SourceSeparator()
        stems = sep.get_available_stems()

        # Order matters per spec — the UI's select_slider uses this list
        # verbatim and the renderer / storage expect lowercase names.
        assert stems == ["vocals", "drums", "bass", "other"], stems

    # ------------------------------------------------------------------
    # is_available — no model load required
    # ------------------------------------------------------------------

    def test_separator_is_available_returns_bool(self) -> None:
        """``is_available()`` must return a real bool, not a truthy proxy.

        ``app.py`` branches on this value with ``if separator.is_available():``
        and ``st.session_state[…] = separator.is_available()`` — both rely
        on the result being a clean boolean.
        """
        from beginner_tab import SourceSeparator

        result = SourceSeparator.is_available()

        assert isinstance(result, bool), (
            f"expected bool, got {type(result).__name__}"
        )

    # ------------------------------------------------------------------
    # Negative path — invalid stem name
    # ------------------------------------------------------------------

    def test_separator_raises_error_on_invalid_stem(self, tmp_path) -> None:
        """``separate()`` rejects unknown stem names with a clear message.

        The validation runs *before* the lazy model load, so this test
        finishes in well under a second — no Demucs inference needed.
        We still pass a real (synthesised) WAV so the call resembles a
        real invocation; the audio is never actually decoded.
        """
        from beginner_tab import SourceSeparator, SourceSeparationError

        # Synthesise a 1-second silent WAV so the path argument is a
        # real, openable audio file.  The validation short-circuits
        # before this is ever read.
        audio_path = tmp_path / "silence.wav"
        sr = 44100
        silent = np.zeros(sr, dtype=np.float32)
        sf.write(audio_path, silent, sr)

        sep = SourceSeparator()
        bad_stem = "invalid_stem"

        with pytest.raises(SourceSeparationError) as exc_info:
            sep.separate(str(audio_path), target_stem=bad_stem)

        # The error message must mention the offending stem name so the
        # user can fix their input without reading the traceback.
        assert bad_stem in str(exc_info.value), (
            f"stem name not in error message: {exc_info.value!r}"
        )

        # And the model must NOT have been loaded as a side effect of
        # the rejected call — the spec promises validation is cheap.
        assert sep._model is None, (
            "model loaded despite invalid-stem rejection — validation "
            "should short-circuit before _load_model"
        )
