# app/memory/session_memory.py

from typing import List, Dict
from .session_store import SessionStore

MAX_TURNS = 6


class SessionMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.store = SessionStore()

        # 🔴 CRITICAL FIX:
        # Ensure session exists but NEVER overwrite existing history
        self.store.create_session(session_id)

        # ✅ Load existing history (if any)
        self.history: List[Dict] = self.store.load_history(session_id)

    # ----------------------------
    # ADD MESSAGES
    # ----------------------------

    def add_user(self, message: str):
        self.history.append({
            "role": "user",
            "content": message
        })
        self._trim()
        self.store.save_history(self.session_id, self.history)

    def add_assistant(self, message: str):
        self.history.append({
            "role": "assistant",
            "content": message
        })
        self._trim()
        self.store.save_history(self.session_id, self.history)

    # ----------------------------
    # CONTEXT FOR PROMPT
    # ----------------------------

    def get_context(self) -> str:
        return "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in self.history
        )

    # ----------------------------
    # TRIM HISTORY (SLIDING WINDOW)
    # ----------------------------

    def _trim(self):
        if len(self.history) > MAX_TURNS * 2:
            self.history = self.history[-MAX_TURNS * 2:]
