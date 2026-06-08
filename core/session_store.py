"""
SQLite-backed session store for RISA.
Provides ChatGPT-like persistent conversation sessions per device.
"""

import os
import sqlite3
import threading
import time
import uuid as _uuid_mod
from typing import List, Optional

# core/ lives one level below project root; chroma_db/ is at project root
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db", "sessions.db")
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT    PRIMARY KEY,
                device_key  TEXT    NOT NULL,
                title       TEXT    NOT NULL DEFAULT 'New Chat',
                created_at  INTEGER NOT NULL,
                updated_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_device
                ON sessions (device_key, updated_at DESC);

            CREATE TABLE IF NOT EXISTS session_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                created_at  INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON session_messages (session_id, created_at);
        """)
        conn.commit()
        conn.close()


def create_session(device_key: str, title: str = "New Chat") -> str:
    session_id = str(_uuid_mod.uuid4())
    now = int(time.time())
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO sessions (id, device_key, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, device_key, title[:120], now, now),
        )
        conn.commit()
        conn.close()
    return session_id


def list_sessions(device_key: str, limit: int = 60) -> List[dict]:
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions "
            "WHERE device_key = ? ORDER BY updated_at DESC LIMIT ?",
            (device_key, limit),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]


def get_session(session_id: str, device_key: str) -> Optional[dict]:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ? AND device_key = ?",
            (session_id, device_key),
        ).fetchone()
        conn.close()
    return dict(row) if row else None


def delete_session(session_id: str, device_key: str) -> bool:
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "DELETE FROM sessions WHERE id = ? AND device_key = ?",
            (session_id, device_key),
        )
        conn.commit()
        conn.close()
    return cur.rowcount > 0


def rename_session(session_id: str, device_key: str, title: str) -> bool:
    with _lock:
        conn = _get_conn()
        cur = conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ? AND device_key = ?",
            (title[:120], int(time.time()), session_id, device_key),
        )
        conn.commit()
        conn.close()
    return cur.rowcount > 0


def add_message(session_id: str, role: str, content: str) -> None:
    now = int(time.time())
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO session_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content[:6000], now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        conn.commit()
        conn.close()


def update_title_if_default(session_id: str, title: str) -> None:
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ? AND title = 'New Chat'",
            (title[:80], session_id),
        )
        conn.commit()
        conn.close()


def get_messages(session_id: str, device_key: str, limit: int = 60) -> List[dict]:
    with _lock:
        conn = _get_conn()
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? AND device_key = ?",
            (session_id, device_key),
        ).fetchone()
        if not exists:
            conn.close()
            return []
        rows = conn.execute(
            "SELECT role, content, created_at FROM session_messages "
            "WHERE session_id = ? ORDER BY created_at LIMIT ?",
            (session_id, limit),
        ).fetchall()
        conn.close()
    return [dict(r) for r in rows]
