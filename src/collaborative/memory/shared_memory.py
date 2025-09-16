import hashlib
import time
from pathlib import Path

from src.collaborative.memory.convergence import ConvergenceChecker
from src.collaborative.memory.storage import SessionStorage


class SharedMemory:
    def __init__(
        self,
        topic: str,
        max_iterations: int = 5,
        min_feedback_threshold: int = 1,
        storage_dir: str = "memory",
    ):
        self.topic = topic
        self.session_id = self._generate_session_id(topic)
        self.storage_dir = Path(storage_dir)

        self.storage = SessionStorage(self.storage_dir, self.session_id)
        self.convergence_checker = ConvergenceChecker(
            max_iterations, min_feedback_threshold
        )

        self.data = self.storage.load_session()
        if not self.data.get("topic"):
            self.data.update(
                {
                    "topic": topic,
                    "session_id": self.session_id,
                    "current_draft": "",
                    "draft_version": 0,
                    "iteration": 0,
                    "feedback_history": [],
                    "current_feedback": [],
                    "converged": False,
                    "convergence_reason": None,
                }
            )
            self._persist()

    def _generate_session_id(self, topic: str) -> str:
        timestamp = str(int(time.time()))
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
        return f"{topic_hash}_{timestamp}"

    def _persist(self) -> None:
        self.storage.save_session(self.data)

    def store_draft(self, draft: str) -> None:
        self.data["current_draft"] = draft
        self.data["draft_version"] += 1
        self._persist()

    def get_current_draft(self) -> str:
        return self.data["current_draft"]

    def add_feedback(self, feedback: list[str]) -> None:
        self.data["current_feedback"].extend(feedback)
        self._persist()

    def get_current_feedback(self) -> list[str]:
        return self.data["current_feedback"].copy()

    def next_iteration(self) -> None:
        self.data["feedback_history"].append(self.data["current_feedback"].copy())
        self.data["current_feedback"] = []
        self.data["iteration"] += 1
        self._persist()

    def check_convergence(self) -> tuple[bool, str]:
        converged, reason = self.convergence_checker.check_convergence(
            self.data["iteration"],
            self.data["current_feedback"],
            self.data["feedback_history"],
        )

        if converged:
            self.data["converged"] = True
            self.data["convergence_reason"] = reason
            self._persist()

        return converged, reason

    def get_session_summary(self) -> dict:
        return self.data.copy()
