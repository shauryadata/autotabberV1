# ==========================================================================
# File:        tab_storage.py
# Authors:     Daniel Ahn, Shauryaditya Singh
# Date:        2026-04-06
# Description: Persistent storage layer for generated guitar tabs using
#              Python's built-in sqlite3 module.  Stores tab text along
#              with metadata (filename, detector used, tempo, max fret,
#              note count, timestamp) in a single SQLite database file.
#              Provides CRUD operations: save, list, get, delete, count.
# ==========================================================================

"""Persistent tab storage backed by SQLite.

Uses Python's built-in ``sqlite3`` module — no extra dependencies.
The database file is created automatically on first use.

Typical usage::

    storage = TabStorage()                   # opens/creates tabs.db
    tab_id  = storage.save("song.mp3", tab_text, tempo=92.0, max_fret=7)
    records = storage.list_tabs()
    record  = storage.get_tab(tab_id)
    storage.delete_tab(tab_id)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


class TabStorageError(Exception):
    """Raised on database errors."""


class TabStorage:
    """Store, retrieve, and manage generated guitar tabs in a SQLite database.

    Each saved tab record contains:

    * ``id``          — auto-incremented primary key.
    * ``filename``    — original audio filename.
    * ``reference_url`` — optional YouTube / reference URL (label only).
    * ``detector``    — pitch-detection backend used (e.g. ``"basic-pitch"``).
    * ``tempo``       — estimated BPM.
    * ``max_fret``    — max-fret setting used.
    * ``one_string``  — whether one-string mode was active.
    * ``note_count``  — number of note/chord events in the tab.
    * ``stem``        — Demucs stem the tab was generated from
      (``"vocals"``, ``"drums"``, ``"bass"``, ``"other"``), or
      ``"full mix"`` when source separation was off / unavailable.
      Added in v0.3 and back-filled to ``"full mix"`` for any rows
      created before the column existed.
    * ``tab_text``    — the full ASCII tab string.
    * ``created_at``  — UTC timestamp string.

    Example::

        db = TabStorage("tabs.db")
        tid = db.save(
            filename="my_song.mp3",
            tab_text=tab_str,
            detector="basic-pitch",
            tempo=98.0,
            max_fret=7,
            one_string=False,
            note_count=132,
        )
        rows = db.list_tabs()          # newest first
        row  = db.get_tab(tid)
        db.delete_tab(tid)
    """

    # Default value for the ``stem`` column.  Stored on every row so the
    # history UI can always show provenance, even for tabs generated
    # before source separation existed.
    DEFAULT_STEM: str = "full mix"

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS tabs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT    NOT NULL,
            reference_url TEXT,
            detector      TEXT,
            tempo         REAL,
            max_fret      INTEGER,
            one_string    INTEGER DEFAULT 0,
            note_count    INTEGER DEFAULT 0,
            stem          TEXT    NOT NULL DEFAULT 'full mix',
            tab_text      TEXT    NOT NULL,
            created_at    TEXT    NOT NULL
        )
    """

    def __init__(self, db_path: str | Path = "tabs.db") -> None:
        """Open (or create) the SQLite database.

        Args:
            db_path: Path to the ``.db`` file.  Defaults to ``tabs.db``
                in the current working directory.
        """
        self.db_path = str(db_path)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        filename: str,
        tab_text: str,
        *,
        reference_url: str | None = None,
        detector: str | None = None,
        tempo: float | None = None,
        max_fret: int | None = None,
        one_string: bool = False,
        note_count: int = 0,
        stem: str | None = None,
    ) -> int:
        """Save a generated tab to the database.

        Args:
            filename: Original audio filename (for display).
            tab_text: Full ASCII tab string.
            reference_url: Optional reference URL (label only).
            detector: Pitch-detection backend name.
            tempo: Estimated BPM.
            max_fret: Max-fret setting used.
            one_string: Whether one-string mode was active.
            note_count: Number of note / chord events.
            stem: Source-separation stem the tab was generated from
                (``"vocals"`` / ``"drums"`` / ``"bass"`` / ``"other"``)
                or ``None`` / unset for tabs generated from the full
                mix.  Stored as :attr:`DEFAULT_STEM` (``"full mix"``)
                when ``None`` so the column is queryable without
                NULL-handling downstream.

        Returns:
            The ``id`` of the newly inserted row.
        """
        created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        # Coalesce ``None`` to the default so reads never have to special-case
        # missing-stem rows.
        stem_value = stem if stem else self.DEFAULT_STEM
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO tabs
                   (filename, reference_url, detector, tempo, max_fret,
                    one_string, note_count, stem, tab_text, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    filename,
                    reference_url,
                    detector,
                    tempo,
                    max_fret,
                    int(one_string),
                    note_count,
                    stem_value,
                    tab_text,
                    created_at,
                ),
            )
            return int(cursor.lastrowid)

    def list_tabs(self) -> list[dict]:
        """Return all tab records, newest first.

        Returns:
            List of dicts with keys matching the ``tabs`` table columns.
            The ``tab_text`` field is excluded to keep the list lightweight.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, filename, reference_url, detector, tempo,
                          max_fret, one_string, note_count, stem, created_at
                   FROM tabs
                   ORDER BY id DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_tab(self, tab_id: int) -> dict | None:
        """Retrieve a single tab record including its ``tab_text``.

        Args:
            tab_id: The ``id`` of the record to retrieve.

        Returns:
            Dict of all column values, or ``None`` if not found.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM tabs WHERE id = ?", (tab_id,)
            ).fetchone()
        return dict(row) if row else None

    def delete_tab(self, tab_id: int) -> None:
        """Delete a tab record permanently.

        Args:
            tab_id: The ``id`` of the record to delete.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM tabs WHERE id = ?", (tab_id,))

    def count(self) -> int:
        """Return the total number of saved tabs."""
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM tabs").fetchone()[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create the ``tabs`` table if it does not already exist, then
        run any column-level migrations needed for older databases.

        Called automatically during ``__init__`` so the database is
        ready to use immediately after construction.
        """
        with self._connect() as conn:
            conn.execute(self._CREATE_TABLE)
            # ---- Migration: add the ``stem`` column on pre-v0.3 DBs ----
            # SQLite's ``ALTER TABLE ADD COLUMN ... DEFAULT '...'`` back-fills
            # every existing row with the default value, so users who upgrade
            # from an older AutoTabber version see "full mix" on their old
            # tabs without any data loss.  We probe ``PRAGMA table_info``
            # rather than catching the duplicate-column OperationalError
            # because the latter would also swallow real schema errors.
            existing = {
                row[1] for row in conn.execute("PRAGMA table_info(tabs)").fetchall()
            }
            if "stem" not in existing:
                conn.execute(
                    f"ALTER TABLE tabs ADD COLUMN stem TEXT NOT NULL "
                    f"DEFAULT '{self.DEFAULT_STEM}'"
                )

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the SQLite database file.

        Returns:
            sqlite3.Connection: An open database connection.

        Raises:
            TabStorageError: If sqlite3 cannot open the file.
        """
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            raise TabStorageError(f"Cannot open database '{self.db_path}': {exc}") from exc
