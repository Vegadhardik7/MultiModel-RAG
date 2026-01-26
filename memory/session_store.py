import sqlite3
import json
from typing import List, Dict
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "chat_sessions.db"


class SessionStore:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH,check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            history TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """)
        self.conn.commit()

    def create_session(self, session_id: str):
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (session_id, json.dumps([]), now, now)
        )
        self.conn.commit()

    def load_history(self, session_id: str) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT history FROM sessions WHERE session_id=?",
            (session_id,)
        )
        row = cur.fetchone()
        if not row:
            return []
        return json.loads(row["history"])

    def save_history(self, session_id: str, history: List[Dict]):
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "UPDATE sessions SET history=?, updated_at=? WHERE session_id=?",
            (json.dumps(history), now, session_id)
        )
        self.conn.commit()

    def list_sessions(self):
        cur = self.conn.execute(
            "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        )
        return [dict(row) for row in cur.fetchall()]

    def delete_session(self, session_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM sessions WHERE session_id=?",
            (session_id,)
        )
        self.conn.commit()
        # sqlite3.Cursor.rowcount may be -1 for some statements; fetch to confirm
        return cur.rowcount != 0
