import hashlib
import time
import uuid
from pathlib import Path

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.collaborative.memory.convergence import ConvergenceChecker
from src.collaborative.memory.storage import SessionStorage
from src.utils.data import Outline, ResearchChunk


@dataclass
class MemoryState:
    """Dataclass for the memory state - provides type safety and clean access."""

    # Session metadata
    topic: str
    session_id: str
    iteration: int = 0
    converged: bool = False
    convergence_reason: Optional[str] = None

    # Current workflow state
    initial_outline: Optional[Outline] = None
    article_content: str = ""

    # Content by iteration
    drafts_by_iteration: Dict[str, str] = field(
        default_factory=dict
    )  # iteration -> full draft content
    article_sections_by_iteration: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )  # iteration -> {section_name: content}

    # Research storage
    research_chunks: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # chunk_id -> chunk data

    # Feedback storage
    structured_feedback: List[Dict[str, Any]] = field(
        default_factory=list
    )  # feedback items with IDs and status

    # Workflow metadata (decisions, timing, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedMemory:
    @classmethod
    def __class_getitem__(cls, item):
        """Make class appear generic to LangGraph type system."""
        return cls

    def __init__(
        self,
        topic: str,
        max_iterations: int = 5,
        min_feedback_threshold: int = 1,
        storage_dir: str = "data/memory",
    ):
        self.topic = topic
        self.session_id = self._generate_session_id(topic)
        self.storage_dir = Path(storage_dir)

        self.storage = SessionStorage(self.storage_dir, self.session_id)
        self.convergence_checker = ConvergenceChecker(
            max_iterations, min_feedback_threshold
        )

        # Load existing data or create new state
        raw_data = self.storage.load_session()
        if raw_data.get("topic"):
            # Convert loaded dict to dataclass
            self.state = MemoryState(**raw_data)
        else:
            # Create new state
            self.state = MemoryState(topic=topic, session_id=self.session_id)

        # Persist if new state
        if not raw_data.get("topic"):
            self._persist()

    @property
    def data(self) -> Dict[str, Any]:
        """Backward compatibility: provide dict access to state data."""
        from dataclasses import asdict

        return asdict(self.state)

    # LangGraph compatibility methods
    def __getitem__(self, key: str):
        """Dictionary-like access for LangGraph compatibility."""
        return getattr(self.state, key)

    def __setitem__(self, key: str, value):
        """Dictionary-like assignment for LangGraph compatibility."""
        setattr(self.state, key, value)
        self._persist()

    def keys(self):
        """Return state field names for LangGraph compatibility."""
        from dataclasses import fields

        return [field.name for field in fields(self.state)]

    def __iter__(self):
        """Iterate over state field names for LangGraph compatibility."""
        return iter(self.keys())

    def items(self):
        """Return key-value pairs for LangGraph compatibility."""
        from dataclasses import fields

        return [
            (field.name, getattr(self.state, field.name))
            for field in fields(self.state)
        ]

    def values(self):
        """Return state field values for LangGraph compatibility."""
        from dataclasses import fields

        return [getattr(self.state, field.name) for field in fields(self.state)]

    def __contains__(self, key):
        """Check if state has field for LangGraph compatibility."""
        return hasattr(self.state, key)

    def get(self, key, default=None):
        """Get state field value with default for LangGraph compatibility."""
        return getattr(self.state, key, default)

    def update(self, updates: Dict[str, Any]):
        """Update multiple state fields for LangGraph compatibility."""
        for key, value in updates.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        self._persist()

    def _generate_session_id(self, topic: str) -> str:
        timestamp = str(int(time.time()))
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
        return f"{topic_hash}_{timestamp}"

    def _persist(self) -> None:
        from dataclasses import asdict

        self.storage.save_session(asdict(self.state))

    def store_draft(self, draft: str, iteration: Optional[int] = None) -> None:
        """Store draft for specific iteration."""
        if iteration is None:
            iteration = self.state.iteration
        self.state.drafts_by_iteration[str(iteration)] = draft
        self._persist()

    def get_current_draft(self) -> str:
        """Get draft for current iteration."""
        current_iteration = str(self.state.iteration)
        return self.state.drafts_by_iteration.get(current_iteration, "")

    def get_current_iteration(self) -> int:
        """Get current iteration number."""
        return self.state.iteration

    def get_draft_by_iteration(self, iteration: int) -> str:
        """Get draft for specific iteration."""
        return self.state.drafts_by_iteration.get(str(iteration), "")

    def next_iteration(self) -> None:
        """Move to next iteration."""
        self.state.iteration += 1
        self._persist()

    def check_convergence(self) -> tuple[bool, str]:
        # Note: This method may need adjustment based on what current_feedback and feedback_history should be
        # For now, using structured_feedback as a proxy
        converged, reason = self.convergence_checker.check_convergence(
            self.state.iteration,
            (
                self.state.structured_feedback[-1]
                if self.state.structured_feedback
                else {}
            ),
            self.state.structured_feedback,
        )

        if converged:
            self.state.converged = True
            self.state.convergence_reason = reason
            self._persist()

        return converged, reason

    def get_session_summary(self) -> dict:
        from dataclasses import asdict

        return asdict(self.state)

    # =================== Research Chunks Management ===================

    def store_research_chunk(self, chunk: ResearchChunk) -> None:
        """Store a ResearchChunk object directly."""
        self.state.research_chunks[chunk.chunk_id] = chunk.to_dict()
        self._persist()

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, ResearchChunk]:
        """Retrieve multiple research chunks by their IDs as ResearchChunk objects."""
        chunks = {}
        for chunk_id in chunk_ids:
            chunk_data = self.state.research_chunks.get(chunk_id)
            if chunk_data:
                # Handle backward compatibility: convert 'summary' to 'description' if needed
                if "summary" in chunk_data and "description" not in chunk_data:
                    chunk_data["description"] = chunk_data.pop("summary")
                chunk_data["chunk_id"] = chunk_id  # Ensure chunk_id is present
                chunks[chunk_id] = ResearchChunk.from_dict(chunk_data)
        return chunks

    def get_chunk_summaries(self) -> Dict[str, str]:
        """Get summaries of all stored chunks for LLM chunk selection."""
        summaries = {}
        for chunk_id, chunk_data in self.state.research_chunks.items():
            description = chunk_data.get("description", "")
            source = chunk_data.get("source", "")
            summary = f"{description}"
            if source:
                summary += f" (Source: {source})"
            summaries[chunk_id] = summary
        return summaries

    def get_all_chunk_ids(self) -> List[str]:
        """Get all available chunk IDs."""
        return list(self.state.research_chunks.keys())

    # =================== Article Sections Management ===================

    def store_article_sections(self, iteration: int, sections: Dict[str, str]) -> None:
        """Store article sections for a specific iteration."""
        self.state.article_sections_by_iteration[str(iteration)] = sections.copy()
        self._persist()

    def get_section_by_iteration(
        self, section_name: str, iteration: int
    ) -> Optional[str]:
        """Get a specific section from a specific iteration."""
        iteration_data = self.state.article_sections_by_iteration.get(
            str(iteration), {}
        )
        return iteration_data.get(section_name)

    def get_sections_from_iteration(self, iteration: int) -> Dict[str, str]:
        """Get all sections from a specific iteration."""
        return self.state.article_sections_by_iteration.get(str(iteration), {}).copy()

    # =================== Structured Feedback Management ===================

    def store_structured_feedback(
        self,
        feedback_text: str,
        iteration: int,
        target_section: Optional[str] = None,
        feedback_id: Optional[str] = None,
    ) -> str:
        """Store structured feedback with ID and status tracking."""
        if feedback_id is None:
            feedback_id = f"feedback_{iteration}_{str(uuid.uuid4())[:8]}"

        feedback_item = {
            "id": feedback_id,
            "text": feedback_text,
            "iteration": iteration,
            "target_section": target_section,
            "status": "pending",
            "created_at": time.time(),
            "applied_at": None,
            "application_notes": "",
        }

        self.state.structured_feedback.append(feedback_item)
        self._persist()

        return feedback_id

    def mark_feedback_applied(self, feedback_id: str, notes: str = "") -> bool:
        """Mark feedback as applied."""
        for feedback in self.state.structured_feedback:
            if feedback["id"] == feedback_id:
                feedback["status"] = "applied"
                feedback["application_notes"] = notes
                feedback["applied_at"] = time.time()
                self._persist()
                return True
        return False

    def mark_feedback_ignored(self, feedback_id: str, notes: str = "") -> bool:
        """Mark feedback as ignored."""
        for feedback in self.state.structured_feedback:
            if feedback["id"] == feedback_id:
                feedback["status"] = "ignored"
                feedback["application_notes"] = notes
                feedback["applied_at"] = time.time()
                self._persist()
                return True
        return False

    def get_feedback_by_id(self, feedback_id: str) -> Optional[Dict]:
        """Get feedback by its ID."""
        for feedback in self.state.structured_feedback:
            if feedback["id"] == feedback_id:
                return feedback.copy()
        return None

    def get_pending_feedback(self) -> List[Dict]:
        """Get all feedback that hasn't been applied yet."""
        return [
            feedback.copy()
            for feedback in self.state.structured_feedback
            if feedback.get("status") == "pending"
        ]

    def get_feedback_for_section(self, section_name: str) -> List[Dict]:
        """Get all feedback that targets a specific section."""
        return [
            feedback.copy()
            for feedback in self.state.structured_feedback
            if feedback.get("target_section") == section_name
        ]

    # =================== Enhanced Draft Management ===================

    def store_draft_with_sections(
        self, draft: str, sections: Optional[Dict[str, str]] = None
    ) -> None:
        """Store draft and extract/store sections."""
        self.store_draft(draft)

        # If sections not provided, try to extract them from the draft
        if sections is None:
            sections = self._extract_sections_from_draft(draft)

        # Store sections for current iteration
        self.store_article_sections(self.state.iteration, sections)

    def _extract_sections_from_draft(self, draft: str) -> Dict[str, str]:
        """Extract sections from draft content based on markdown headers."""
        sections = {}
        lines = draft.split("\n")
        current_section = None
        current_content = []

        for line in lines:
            # Check for markdown headers (## Section Name)
            if line.strip().startswith("## "):
                # Save previous section if exists
                if current_section and current_content:
                    sections[current_section] = "\n".join(current_content).strip()

                # Start new section
                current_section = line.replace("##", "").strip()
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section and current_content:
            sections[current_section] = "\n".join(current_content).strip()

        return sections
