# app/memory/session_store.py

import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# --------------------------------------------------
# DATABASE PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / "data" / "chat_sessions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# SINGLETON CONNECTION
# --------------------------------------------------

_CONN: Optional[sqlite3.Connection] = None


def _get_connection() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False
        )
        _CONN.row_factory = sqlite3.Row
    return _CONN


# --------------------------------------------------
# SESSION STORE
# --------------------------------------------------

class SessionStore:
    """
    Persistent storage for chat sessions.
    - One row per session
    - History stored as JSON
    """

    def __init__(self):
        self.conn = _get_connection()
        self._init_tables()

    # --------------------------------------------------

    def _init_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                history TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    # --------------------------------------------------
    # SESSION LIFECYCLE
    # --------------------------------------------------

    def create_session(self, session_id: str) -> None:
        """
        Create a session ONLY if it does not already exist.
        Never overwrites existing history.
        """
        now = time.time()

        self.conn.execute(
            """
            INSERT OR IGNORE INTO sessions
            (session_id, history, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, json.dumps([]), now, now)
        )
        self.conn.commit()

    # --------------------------------------------------

    def delete_session(self, session_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    # --------------------------------------------------
    # HISTORY
    # --------------------------------------------------

    def load_history(self, session_id: str) -> List[Dict]:
        """
        Load chat history for a session.
        If session does not exist, returns empty list
        WITHOUT creating or mutating anything.
        """
        cur = self.conn.execute(
            "SELECT history FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return []

        try:
            return json.loads(row["history"])
        except Exception:
            return []

    # --------------------------------------------------

    def save_history(self, session_id: str, history: List[Dict]) -> None:
        """
        Persist full history for a session.
        """
        now = time.time()

        self.conn.execute(
            """
            UPDATE sessions
            SET history = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (json.dumps(history), now, session_id)
        )
        self.conn.commit()

    # --------------------------------------------------
    # DEBUG / ADMIN
    # --------------------------------------------------

    def list_sessions(self) -> List[str]:
        cur = self.conn.execute(
            "SELECT session_id FROM sessions ORDER BY created_at DESC"
        )
        return [row["session_id"] for row in cur.fetchall()]
