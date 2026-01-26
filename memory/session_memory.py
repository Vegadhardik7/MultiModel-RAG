from typing import List, Dict
from .session_store import SessionStore


MAX_TURNS = 6


class SessionMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.store = SessionStore()
        self.history: List[Dict] = self.store.load_history(session_id)

    def add_user(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim()
        self.store.save_history(self.session_id, self.history)

    def add_assistant(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._trim()
        self.store.save_history(self.session_id, self.history)

    def get_context(self) -> str:
        return "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in self.history
        )

    def _trim(self):
        if len(self.history) > MAX_TURNS * 2:
            self.history = self.history[-MAX_TURNS * 2:]
