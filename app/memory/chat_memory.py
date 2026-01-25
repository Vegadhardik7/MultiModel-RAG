from typing import List, Dict

MAX_TURNS = 4  # last 4 user+assistant turns


class ChatMemory:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add_user(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def get_context(self) -> str:
        """
        Convert history into plain text context.
        """
        lines = []
        for turn in self.history:
            role = turn["role"].capitalize()
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def _trim(self):
        """
        Keep memory bounded.
        """
        if len(self.history) > MAX_TURNS * 2:
            self.history = self.history[-MAX_TURNS * 2:]
