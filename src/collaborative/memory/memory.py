import hashlib
import time
import uuid
from pathlib import Path

import logging
from typing import Any, Dict, Iterable, List, Optional, TypedDict

logger = logging.getLogger(__name__)

from src.collaborative.memory.convergence import ConvergenceChecker
from src.collaborative.memory.storage import SessionStorage
from src.collaborative.theory_of_mind import TheoryOfMindModule
from src.utils.data import Outline, ResearchChunk, ReviewFeedback


class MemoryState(TypedDict, total=False):
    """TypedDict for LangGraph compatibility - provides both type safety and direct serialization."""

    # Session metadata (required fields)
    topic: str
    session_id: str

    # Optional fields with defaults
    iteration: int
    converged: bool
    convergence_reason: Optional[str]

    # Current workflow state
    initial_outline: Optional[Outline]

    # Content by iteration
    drafts_by_iteration: Dict[str, str]  # iteration -> full draft content
    article_sections_by_iteration: Dict[
        str, Dict[str, str]
    ]  # iteration -> {section_name: content}

    # Research storage
    research_chunks: Dict[str, Dict[str, Any]]  # chunk_id -> serialized chunk data
    search_summaries: Dict[
        str, Dict[str, Any]
    ]  # query -> search results with summaries

    # Feedback storage
    structured_feedback: List[Dict[str, Any]]  # serialized feedback items

    # Workflow metadata (decisions, timing, etc.)
    metadata: Dict[str, Any]


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
        tom_enabled: bool = False,
    ):
        self.topic = topic
        self.session_id = self._generate_session_id(topic)
        self.storage_dir = Path(storage_dir)

        self.storage = SessionStorage(self.storage_dir, self.session_id)
        self.convergence_checker = ConvergenceChecker(
            max_iterations, min_feedback_threshold, feedback_addressed_threshold=0.9
        )

        # Initialize Theory of Mind module
        self.tom_module = TheoryOfMindModule(enabled=tom_enabled)

        # Load existing data or create new state
        raw_data = self.storage.load_session()
        if raw_data.get("topic"):
            # Use existing data as TypedDict
            self.state: MemoryState = raw_data
        else:
            # Create new state with defaults
            self.state: MemoryState = {
                "topic": topic,
                "session_id": self.session_id,
                "iteration": 0,
                "converged": False,
                "convergence_reason": None,
                "initial_outline": None,
                "drafts_by_iteration": {},
                "article_sections_by_iteration": {},
                "research_chunks": {},
                "search_summaries": {},
                "structured_feedback": [],
                "metadata": {},
            }

        # Persist if new state
        if not raw_data.get("topic"):
            self._persist()

    @property
    def data(self) -> Dict[str, Any]:
        """Backward compatibility: provide dict access to state data."""
        return dict(self.state)

    # LangGraph compatibility methods
    def __getitem__(self, key: str):
        """Dictionary-like access for LangGraph compatibility."""
        return self.state[key]

    def __setitem__(self, key: str, value):
        """Dictionary-like assignment for LangGraph compatibility."""
        self.state[key] = value
        self._persist()

    def keys(self):
        """Return state field names for LangGraph compatibility."""
        return self.state.keys()

    def __iter__(self):
        """Iterate over state field names for LangGraph compatibility."""
        return iter(self.state)

    def items(self):
        """Return key-value pairs for LangGraph compatibility."""
        return self.state.items()

    def values(self):
        """Return state field values for LangGraph compatibility."""
        return self.state.values()

    def __contains__(self, key):
        """Check if state has field for LangGraph compatibility."""
        return key in self.state

    def get(self, key, default=None):
        """Get state field value with default for LangGraph compatibility."""
        return self.state.get(key, default)

    def update(self, updates: Dict[str, Any]):
        """Update multiple state fields for LangGraph compatibility."""
        self.state.update(updates)
        self._persist()

    def _generate_session_id(self, topic: str) -> str:
        timestamp = str(int(time.time()))
        topic_hash = hashlib.md5(topic.encode()).hexdigest()[:8]
        return f"{topic_hash}_{timestamp}"

    def _persist(self) -> None:
        # Convert state to JSON-serializable format
        serializable_state = self._make_serializable(dict(self.state))
        self.storage.save_session(serializable_state)

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For other objects, try to convert to string
            return str(obj)

    def store_draft(self, draft: str, iteration: Optional[int] = None) -> None:
        """Store draft for specific iteration."""
        if iteration is None:
            iteration = self.state["iteration"]
        self.state["drafts_by_iteration"][str(iteration)] = draft
        self._persist()

    def get_current_draft(self) -> str:
        """Get draft for current iteration."""
        current_iteration = str(self.state["iteration"])
        return self.state["drafts_by_iteration"].get(current_iteration, "")

    def get_current_article(self) -> Optional[Any]:
        """Get current article as Article object."""
        from src.utils.data import Article

        current_draft = self.get_current_draft()
        if not current_draft:
            return None

        current_sections = self.get_sections_from_iteration(self.state["iteration"])

        # Create Article object
        article = Article(
            title=self.state.get("topic", "Untitled"),
            content=current_draft,
            sections=current_sections,
            metadata=self.state.get("metadata", {}),
        )
        return article

    def get_current_iteration(self) -> int:
        """Get current iteration number."""
        return self.state["iteration"]

    def next_iteration(self) -> None:
        """Move to next iteration."""
        old_iteration = self.state["iteration"]
        self.state["iteration"] += 1
        logger.info(
            f"🔍 ITERATION DEBUG: Moving from iteration {old_iteration} to {self.state['iteration']}"
        )
        self._persist()

    def check_convergence(self) -> tuple[bool, str]:
        # Get feedback from the current iteration (the one just completed by reviewer)
        # For iteration 0, we check iteration 0. For iteration 1+, we check current iteration.
        current_iteration = self.state["iteration"]
        current_feedback = [
            fb
            for fb in self.state["structured_feedback"]
            if fb.get("iteration") == current_iteration
            and fb.get("status") == "pending"
        ]

        # DEBUG: Log convergence check details
        logger.info(f"🔍 CONVERGENCE DEBUG: iteration={current_iteration}")
        logger.info(
            f"🔍 CONVERGENCE DEBUG: total structured_feedback={len(self.state['structured_feedback'])}"
        )
        logger.info(
            f"🔍 CONVERGENCE DEBUG: current_feedback for iteration {current_iteration}={len(current_feedback)}"
        )
        if self.state["structured_feedback"]:
            for i, fb in enumerate(self.state["structured_feedback"]):
                logger.info(
                    f"🔍 CONVERGENCE DEBUG: feedback[{i}]: iteration={fb.get('iteration')}, status={fb.get('status')}"
                )

        converged, reason = self.convergence_checker.check_convergence(
            self.state["iteration"],
            current_feedback,
            self.state["structured_feedback"],
            structured_feedback=self.state["structured_feedback"],  # Pass for 90% rule
        )

        if converged:
            self.state["converged"] = True
            self.state["convergence_reason"] = reason
            self._persist()

        return converged, reason

    def get_session_summary(self) -> dict:
        return dict(self.state)

    # =================== Research Chunks Management ===================

    def store_research_chunks(
        self, chunks: Iterable["ResearchChunk"]
    ) -> List[Dict[str, Any]]:
        # Materialize once (in case `chunks` is a generator)
        chunk_list = list(chunks)

        # Ensure the bucket exists
        store = self.state.setdefault("research_chunks", {})

        # Batch serialize + store
        store.update({c.chunk_id: c.to_dict() for c in chunk_list})

        # Single persist for atomicity
        self._persist()

        # Build summaries for agent context
        summaries: List[Dict[str, Any]] = []
        for i, c in enumerate(chunk_list, start=1):
            summary = {
                "chunk_id": c.chunk_id,
                "description": c.description,
                "source": c.source,
                "relevance_rank": i,
            }
            score = (c.metadata or {}).get("relevance_score")
            if score is not None:
                summary["relevance_score"] = score
            summaries.append(summary)

        return summaries

    def get_stored_chunks(self) -> List[ResearchChunk]:
        """Retrieve all stored research chunks."""
        chunks = []
        for chunk_data in self.state["research_chunks"].values():
            if isinstance(chunk_data, dict):
                chunks.append(ResearchChunk.model_validate(chunk_data))
            else:
                # Already a ResearchChunk object
                chunks.append(chunk_data)
        return chunks

    def store_search_summary(self, query: str, search_results: Dict[str, Any]) -> None:
        """Store search results with summaries for LLM context."""
        self.state["search_summaries"][query] = search_results
        self._persist()

    def get_search_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get all search summaries."""
        return self.state["search_summaries"]

    def get_all_chunk_ids(self) -> List[str]:
        """Get all available chunk IDs."""
        return list(self.state["research_chunks"].keys())

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, ResearchChunk]:
        """Get specific chunks by their IDs as typed objects."""
        chunks = {}
        for chunk_id in chunk_ids:
            if chunk_id in self.state["research_chunks"]:
                chunk_data = self.state["research_chunks"][chunk_id]
                chunks[chunk_id] = ResearchChunk.model_validate(chunk_data)
        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[ResearchChunk]:
        """Get a single research chunk by ID as a typed object."""
        chunk_data = self.state["research_chunks"].get(chunk_id)
        if not chunk_data:
            return None
        if isinstance(chunk_data, dict):
            return ResearchChunk.model_validate(chunk_data)
        return chunk_data

    # =================== Typed Feedback Management ===================

    def store_review_feedback(self, feedback: ReviewFeedback) -> None:
        """Store typed review feedback."""
        self.state["structured_feedback"].append(feedback.to_dict())
        self._persist()

    def get_review_feedback(self) -> List[ReviewFeedback]:
        """Get all review feedback as typed objects."""
        feedback_list = []
        for feedback_data in self.state["structured_feedback"]:
            if isinstance(feedback_data, dict) and "overall_score" in feedback_data:
                feedback_list.append(ReviewFeedback.model_validate(feedback_data))
        return feedback_list

    # =================== Article Sections Management ===================

    def store_article_sections(self, iteration: int, sections: Dict[str, str]) -> None:
        """Store article sections for a specific iteration."""
        self.state["article_sections_by_iteration"][str(iteration)] = sections.copy()
        self._persist()

    def get_section_by_iteration(
        self, section_name: str, iteration: int
    ) -> Optional[str]:
        """Get a specific section from a specific iteration."""
        iteration_data = self.state["article_sections_by_iteration"].get(
            str(iteration), {}
        )
        return iteration_data.get(section_name)

    def get_sections_from_iteration(self, iteration: int) -> Dict[str, str]:
        """Get all sections from a specific iteration."""
        return (
            self.state["article_sections_by_iteration"].get(str(iteration), {}).copy()
        )

    # =================== Structured Feedback Management ===================

    def _format_feedback_item(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of feedback with normalized fields."""
        item = feedback.copy()
        base_text = item.get("text", "")
        item.setdefault("feedback", base_text)
        item.setdefault("priority", "medium")
        item.setdefault("target_section", "general")
        return item

    def store_structured_feedback(
        self,
        feedback_text: str,
        iteration: int,
        target_section: Optional[str] = None,
        feedback_id: Optional[str] = None,
        priority: str = "medium",
    ) -> str:
        """Store structured feedback with ID and status tracking."""
        if feedback_id is None:
            feedback_id = f"feedback_{iteration}_{str(uuid.uuid4())[:8]}"

        feedback_item = {
            "id": feedback_id,
            "text": feedback_text,
            "feedback": feedback_text,
            "iteration": iteration,
            "target_section": target_section,
            "status": "pending",
            "created_at": time.time(),
            "priority": priority,
            # Writer's claim tracking
            "writer_claim_status": None,  # claimed_addressed, contested, etc.
            "writer_reasoning": "",
            "writer_claimed_at": None,
            # Reviewer's evaluation tracking
            "reviewer_verification": None,  # verified_addressed, partially_addressed, misunderstood, etc.
            "reviewer_reasoning": "",
            "reviewer_verified_at": None,
            # Legacy fields (for backward compatibility)
            "applied_at": None,
            "application_notes": "",
        }

        self.state["structured_feedback"].append(feedback_item)
        self._persist()

        return feedback_id

    def mark_feedback_claimed_by_writer(
        self,
        feedback_id: str,
        claim_status: str = "claimed_addressed",
        reasoning: str = "",
    ) -> bool:
        """Writer marks feedback as claimed to be addressed with reasoning."""
        for feedback in self.state["structured_feedback"]:
            if feedback["id"] == feedback_id:
                feedback["writer_claim_status"] = claim_status
                feedback["writer_reasoning"] = reasoning
                feedback["writer_claimed_at"] = time.time()
                feedback["status"] = "writer_claimed"  # Intermediate status
                self._persist()
                return True
        return False

    def mark_feedback_verified_by_reviewer(
        self, feedback_id: str, verification_status: str, reasoning: str = ""
    ) -> bool:
        """Reviewer evaluates writer's claim and marks verification status."""
        for feedback in self.state["structured_feedback"]:
            if feedback["id"] == feedback_id:
                feedback["reviewer_verification"] = verification_status
                feedback["reviewer_reasoning"] = reasoning
                feedback["reviewer_verified_at"] = time.time()

                # Update final status based on reviewer's evaluation
                if verification_status == "verified_addressed":
                    feedback["status"] = "applied"
                elif verification_status in ["partially_addressed", "misunderstood"]:
                    feedback["status"] = "partially_applied"
                else:
                    feedback["status"] = "pending"  # Back to pending for re-work

                self._persist()
                return True
        return False

    def mark_feedback_applied(self, feedback_id: str, notes: str = "") -> bool:
        """Legacy method - mark feedback as applied (for backward compatibility)."""
        for feedback in self.state["structured_feedback"]:
            if feedback["id"] == feedback_id:
                feedback["status"] = "applied"
                feedback["application_notes"] = notes
                feedback["applied_at"] = time.time()
                self._persist()
                return True
        return False

    def mark_feedback_ignored(self, feedback_id: str, notes: str = "") -> bool:
        """Mark feedback as ignored."""
        for feedback in self.state["structured_feedback"]:
            if feedback["id"] == feedback_id:
                feedback["status"] = "ignored"
                feedback["application_notes"] = notes
                feedback["applied_at"] = time.time()
                self._persist()
                return True
        return False

    def get_feedback_by_id(self, feedback_id: str) -> Optional[Dict]:
        """Get feedback by its ID."""
        for feedback in self.state["structured_feedback"]:
            if feedback["id"] == feedback_id:
                return self._format_feedback_item(feedback)
        return None

    def contest_feedback(self, feedback_id: str, contest_reasoning: str) -> bool:
        """Writer contests reviewer feedback instead of applying/ignoring it."""
        for feedback in self.state["structured_feedback"]:
            if feedback["id"] == feedback_id:
                feedback["writer_claim_status"] = "contested"
                feedback["writer_reasoning"] = contest_reasoning
                feedback["writer_claimed_at"] = time.time()
                feedback["status"] = "contested"
                feedback["contest_history"] = feedback.get("contest_history", [])
                feedback["contest_history"].append(
                    {
                        "action": "writer_contested",
                        "reasoning": contest_reasoning,
                        "timestamp": time.time(),
                        "iteration": self.state["iteration"],
                    }
                )
                self._persist()
                return True
        return False

    def resolve_contested_feedback(
        self, feedback_id: str, resolution: str, resolver: str, reasoning: str = ""
    ) -> bool:
        """
        Resolve contested feedback.

        Args:
            feedback_id: ID of the contested feedback
            resolution: 'accept_writer_position', 'maintain_reviewer_position', 'compromise'
            resolver: 'writer', 'reviewer', or 'system'
            reasoning: Explanation for the resolution
        """
        for feedback in self.state["structured_feedback"]:
            if feedback["id"] == feedback_id and feedback["status"] == "contested":
                feedback["contest_history"] = feedback.get("contest_history", [])
                feedback["contest_history"].append(
                    {
                        "action": f"resolved_by_{resolver}",
                        "resolution": resolution,
                        "reasoning": reasoning,
                        "timestamp": time.time(),
                        "iteration": self.state["iteration"],
                    }
                )

                # Update final status based on resolution
                if resolution == "accept_writer_position":
                    feedback["status"] = "ignored"  # Writer's disagreement accepted
                elif resolution == "maintain_reviewer_position":
                    feedback["status"] = (
                        "pending"  # Back to pending for writer to address
                    )
                elif resolution == "compromise":
                    feedback["status"] = "partially_applied"  # Compromise reached
                else:
                    feedback["status"] = (
                        "contested"  # Remains contested if no valid resolution
                    )

                feedback["resolved_at"] = time.time()
                feedback["resolved_by"] = resolver
                self._persist()
                return True
        return False

    def get_pending_feedback(self) -> List[Dict]:
        """Get all feedback that hasn't been applied yet."""
        return [
            self._format_feedback_item(feedback)
            for feedback in self.state["structured_feedback"]
            if feedback.get("status") == "pending"
        ]

    def get_feedback_for_section(self, section_name: str) -> List[Dict]:
        """Get all feedback that targets a specific section."""
        return [
            self._format_feedback_item(feedback)
            for feedback in self.state["structured_feedback"]
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
        self.store_article_sections(self.state["iteration"], sections)

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

    def update_article_state(self, article: Any) -> None:
        """
        Centralized method to update article state consistently.

        This is the ONLY method that should be used to update article content
        to avoid state inconsistencies. Agents should return results and let
        this method handle all state updates.

        Args:
            article: Article object with content, sections, and metadata
        """
        # Store draft for persistence
        logger.info(
            f"🔍 STORE DEBUG: Storing draft in iteration {self.state['iteration']} (length: {len(article.content)})"
        )
        self.store_draft(article.content, self.state["iteration"])

        # Update sections for current iteration
        if hasattr(article, "sections") and article.sections:
            self.store_article_sections(self.state["iteration"], article.sections)
        else:
            # Extract sections if not provided
            sections = self._extract_sections_from_draft(article.content)
            self.store_article_sections(self.state["iteration"], sections)

        # Update metadata if provided
        if hasattr(article, "metadata") and article.metadata:
            self.state["metadata"].update(article.metadata)

        # Persist changes
        self._persist()
