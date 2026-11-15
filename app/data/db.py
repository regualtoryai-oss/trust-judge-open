"""SQLite connections and the versioned, repeatable foundation migration.

Connections use autocommit; wrap multi-statement writes in ``transaction``.
The transaction helper nests through savepoints and never swallows failures.
"""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from uuid import uuid4

SCHEMA_VERSION = 1


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    nested = conn.in_transaction
    savepoint = f"sp_{uuid4().hex}"
    conn.execute(f"SAVEPOINT {savepoint}" if nested else "BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        if nested:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        else:
            conn.rollback()
        raise
    else:
        conn.execute(f"RELEASE SAVEPOINT {savepoint}") if nested else conn.commit()


def initialize(path: Path | str) -> None:
    """Apply schema v1 once, without modifying existing immutable sources."""
    conn = connect(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise ValueError(f"database schema {version} is newer than {SCHEMA_VERSION}")
        if version == SCHEMA_VERSION:
            return
        # Serialize competing initializers before rechecking the schema version.
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION:
            conn.commit()
            return
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='topics'").fetchone():
            raise ValueError("unversioned database already contains tables; use an explicit migration")
        sql = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8-sig")
        # executescript implicitly commits. Execute complete SQL statements on the
        # held write transaction instead, including multi-line trigger bodies.
        pending = ""
        for line in sql.splitlines(keepends=True):
            if line.lstrip().upper().startswith("PRAGMA "):
                continue
            pending += line
            if sqlite3.complete_statement(pending):
                conn.execute(pending)
                pending = ""
        if pending.strip() and not all(line.lstrip().startswith("--") for line in pending.splitlines() if line.strip()):
            raise ValueError("incomplete schema statement")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
