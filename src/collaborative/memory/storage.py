import tempfile
from pathlib import Path

import json


class SessionStorage:
    def __init__(self, storage_dir: Path, session_id: str):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = storage_dir / f"{session_id}.json"

    def load_session(self) -> dict:
        if not self.file_path.exists():
            return {
                "topic": "",
                "session_id": "",
                "current_draft": "",
                "draft_version": 0,
                "iteration": 0,
                "feedback_history": [],
                "current_feedback": [],
            }

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return self.load_session()

    def save_session(self, data: dict) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=self.storage_dir,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2, ensure_ascii=False)
            tmp_file.flush()

        Path(tmp_file.name).replace(self.file_path)
