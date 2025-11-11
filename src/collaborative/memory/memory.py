from pathlib import Path

import logging
from typing import Any, Dict, Iterable, List, Optional, TypedDict

from src.collaborative.memory.storage import SessionStorage
from src.collaborative.tom.theory_of_mind import TheoryOfMindModule
from src.collaborative.utils.models import (
    FeedbackStatus,
    FeedbackStoredModel,
    SearchSummary,
    WriterStatusUpdate,
)
from src.collaborative.utils.writer_utils import build_full_article_content
from src.utils.data import Outline, ResearchChunk
from src.utils.data.models import Article

logger = logging.getLogger(__name__)


class MemoryState(TypedDict):
    """TypedDict for LangGraph compatibility - provides both type safety and direct serialization."""

    # Session metadata (required fields)
    topic: str
    session_id: str
    # Required fields with defaults
    iteration: int
    research_chunks: Dict[str, Dict[str, Any]]  # chunk_id -> serialized chunk data
    search_summaries: Dict[
        str, Dict[str, Any]
    ]  # query -> search results with summaries
    # Structure: feedback_items[(section, iteration, item_id)] = item_data
    current_draft: str
    drafts_by_iteration: Dict[str, str]  # iteration -> full draft content
    current_sections: Dict[str, str]  # section_name -> content
    article_sections_by_iteration: Dict[
        str, Dict[str, str]
    ]  # iteration -> {section_name: content}
    feedback_items: Dict[str, FeedbackStoredModel]
    feedback_by_iteration: Dict[
        int, Dict[str, FeedbackStoredModel]
    ]  # iteration -> {item_id: item_data}
    item_index: Dict[str, int]  # item_id -> index (for feedback management)
    suggested_queries_by_iteration: Dict[
        int, List[str]
    ]  # iteration -> list of suggested queries from reviewer

    # Optional fields
    initial_outline: Optional[Outline]
    metadata: Dict[str, Any]
    penalized_chunk_ids: List[str]


class SharedMemory:
    @classmethod
    def __class_getitem__(cls, item):
        """Make class appear generic to LangGraph type system."""
        return cls

    def __init__(
        self,
        topic: str,
        storage_dir: Optional[str] = None,
        tom_enabled: bool = False,
        experiment_name: Optional[str] = None,
    ):
        self.topic = topic
        self.experiment_name = experiment_name
        self.session_id = self._generate_session_id(topic, experiment_name)

        # If no storage_dir provided, use default data/memory
        if storage_dir is None:
            storage_dir = "data/memory"

        # Create memory subdirectory within the storage path
        self.storage_dir = Path(storage_dir) / "memory"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.storage = SessionStorage(self.storage_dir, self.session_id)

        # Initialize Theory of Mind module
        self.tom_module = TheoryOfMindModule(enabled=tom_enabled)

        # Load existing data or create new state
        raw_data = self.storage.load_session()

        # Always initialize optional fields with empty structures if missing
        def ensure_key(d, key, default):
            if key not in d or d[key] is None:
                d[key] = default

        if raw_data.get("topic"):
            # Use existing data as TypedDict
            ensure_key(raw_data, "drafts_by_iteration", {})
            ensure_key(raw_data, "article_sections_by_iteration", {})
            ensure_key(raw_data, "research_chunks", {})
            ensure_key(raw_data, "search_summaries", {})
            ensure_key(raw_data, "feedback_items", {})
            ensure_key(raw_data, "metadata", {})
            ensure_key(raw_data, "item_index", {})
            ensure_key(raw_data, "penalized_chunk_ids", [])
            ensure_key(raw_data, "suggested_queries_by_iteration", {})
            self.state: MemoryState = MemoryState(**raw_data)
        else:
            # Create new state with defaults
            self.state: MemoryState = MemoryState(
                topic=topic,
                session_id=self.session_id,
                iteration=0,
                research_chunks={},
                search_summaries={},
                feedback_items={},
                article_sections_by_iteration={},
                drafts_by_iteration={},
                current_draft="",
                current_sections={},
                initial_outline=None,
                metadata={},
                feedback_by_iteration={},
                item_index={},
                penalized_chunk_ids=[],
                suggested_queries_by_iteration={},
            )

        # Persist if new state
        if not raw_data.get("topic"):
            self._persist()

    # region LangGraph compatibility methods
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
        self.state.update(MemoryState(**updates))
        self._persist()

    # endregion LangGraph compatibility methods

    # =================== Getters for State Fields ===================
    # region Getters for State Fields

    def get_topic(self) -> str:
        """Get the topic of the session."""
        return self.state["topic"]

    def get_outline(self) -> Optional[Outline]:
        """Get the initial outline of the session, if available."""
        return self.state.get("initial_outline")

    def get_session_id(self) -> str:
        """Get the session ID."""
        return self.state["session_id"]

    def get_iteration(self) -> int:
        """Get the current iteration number."""
        return self.state["iteration"]

    def get_search_summaries(self) -> Dict[str, SearchSummary]:
        """Get all search summaries as Pydantic models."""
        summaries = {}
        for search_id, data in self.state["search_summaries"].items():
            try:
                # Use the updated SearchSummary model for validation
                summaries[search_id] = SearchSummary.model_validate(data)
            except Exception as e:
                # Use search_id in the log for clarity
                logger.warning(
                    f"Invalid SearchSummary for search_id '{search_id}' skipped: {e}"
                )
        return summaries

    def get_current_draft(self) -> str:
        """Get the current full draft content."""
        return self.state["current_draft"]

    def get_current_sections(self) -> Dict[str, str]:
        """Get the current sections content."""
        return self.state["current_sections"]

    def get_previous_draft(self) -> Optional[str]:
        """Get draft from previous iteration, if available."""
        prev_iteration = self.state.get("iteration", 0) - 1
        if prev_iteration < 0:
            return None
        return self.state.get("drafts_by_iteration", {}).get(str(prev_iteration))

    def get_current_draft_as_article(self) -> Optional[Article]:
        """Get current article as Article object, rebuilding content from sections if available."""
        from src.collaborative.utils.writer_utils import build_full_article_content

        topic = self.state.get("topic", "Untitled")
        sections = self.state.get("current_sections", {})
        content = self.state.get("current_draft", "")

        # Rebuild content from sections to ensure proper heading structure
        if sections:
            content = build_full_article_content(topic, sections)

        article = Article(
            title=topic,
            content=content,
            sections=sections,
            metadata=self.state.get("metadata", {}),
        )
        return article

    def get_drafts_by_iteration(self, iteration: int) -> Dict[str, str]:
        """Get all drafts from a specific iteration."""
        value = self.state["drafts_by_iteration"].get(str(iteration), {})
        if isinstance(value, dict):
            return value
        return {}

    def get_sections_by_iteration(self, iteration: int) -> Dict[str, str]:
        """Get all sections from a specific iteration."""
        return (
            self.state["article_sections_by_iteration"].get(str(iteration), {}).copy()
        )

    # endregion Getters for State Fields

    # =================== Session Management ===================
    # region Session Management
    def _generate_session_id(
        self, topic: str, experiment_name: Optional[str] = None
    ) -> str:
        """Generate session ID matching article naming: {method}_{topic}."""
        # Sanitize topic for filesystem use
        safe_topic = topic.replace("/", "_").replace(" ", "_")
        # Remove any other problematic characters
        import re

        safe_topic = re.sub(r"[^\w\-_]", "_", safe_topic)
        # Collapse multiple underscores
        safe_topic = re.sub(r"_+", "_", safe_topic).strip("_")

        # Use experiment_name (method name) as prefix if provided
        if experiment_name:
            return f"{experiment_name}_{safe_topic}"
        return safe_topic

    def _persist(self) -> None:
        # Convert state to JSON-serializable format
        serializable_state = self._make_serializable(dict(self.state))
        if not isinstance(serializable_state, dict):
            serializable_state = dict(self.state)
        self.storage.save_session(serializable_state)

    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For other objects, try to convert to string
            return str(obj)

    def next_iteration(self) -> None:
        """Move to next iteration."""
        old_iteration = self.state.get("iteration", 0)
        self.state["iteration"] = old_iteration + 1
        logger.info(
            f"Moving from iteration {old_iteration} to {self.state['iteration']}"
        )
        self._persist()

    # endregion Session Management

    # =================== Research Chunks Management ===================
    # region Research Chunks Management
    def store_chunks(self, chunks: Iterable["ResearchChunk"]) -> List[Dict[str, Any]]:
        # Materialize once (in case `chunks` is a generator)
        chunk_list = list(chunks)

        # Ensure the bucket exists
        store = self.state.setdefault("research_chunks", {})

        # Batch serialize + store
        store.update({c.chunk_id: c.model_dump() for c in chunk_list})

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
        """Retrieve all stored research chunks, skipping invalid ones."""
        chunks = []
        for chunk_data in self.state["research_chunks"].values():
            if isinstance(chunk_data, dict):
                try:
                    chunks.append(ResearchChunk.model_validate(chunk_data))
                except Exception as e:
                    logger.warning(f"Invalid ResearchChunk data skipped: {e}")
            else:
                # Already a ResearchChunk object
                chunks.append(chunk_data)
        return chunks

    def store_search_summary(
        self, search_id: str, search_results: Dict[str, Any]
    ) -> None:
        """
        Store search results with summaries for LLM context.
        Uses the 'search_id' as the storage key and stores 'source_queries' list explicitly.
        """
        # 1. Ensure required fields exist in the input dictionary
        source_queries = search_results.get("source_queries")
        if not isinstance(source_queries, list):
            raise ValueError("Search results must contain a 'source_queries' list.")

        # 2. Instantiate the updated Pydantic model
        summary = SearchSummary(
            search_id=search_id,  # The key used for storage
            source_queries=source_queries,  # The list of input queries
            results=search_results.get("results", []),  # Chunk summaries
            metadata=search_results.get("metadata", {}),
        )

        # 3. Store using the flexible search_id key
        self.state["search_summaries"][search_id] = summary.model_dump()
        self._persist()

    def get_all_chunk_ids(self) -> List[str]:
        """Get all available chunk IDs."""
        return list(self.state["research_chunks"].keys())

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, ResearchChunk]:
        """Get specific chunks by their IDs as typed objects, skipping invalid ones."""
        chunks = {}
        for chunk_id in chunk_ids:
            if chunk_id in self.state["research_chunks"]:
                chunk_data = self.state["research_chunks"][chunk_id]
                try:
                    chunks[chunk_id] = ResearchChunk.model_validate(chunk_data)
                except Exception as e:
                    logger.warning(
                        f"Invalid ResearchChunk for id {chunk_id} skipped: {e}"
                    )
        return chunks

    # endregion Research Chunks Management

    def delete_chunks_by_ids(self, chunk_ids: List[str]) -> int:
        """Delete specified chunks and purge references from search summaries.

        Returns the number of removed chunk records.
        """
        if not chunk_ids:
            return 0

        # Remove from research_chunks
        removed = 0
        store = self.state.setdefault("research_chunks", {})
        for cid in list(chunk_ids):
            if cid in store:
                del store[cid]
                removed += 1

        # Purge from search_summaries results lists
        summaries = self.state.setdefault("search_summaries", {})
        for sid, summary in summaries.items():
            try:
                results = (
                    summary.get("results", []) if isinstance(summary, dict) else []
                )
                if results:
                    filtered = [
                        r for r in results if r.get("chunk_id") not in chunk_ids
                    ]
                    # Only assign back if anything changed to avoid unnecessary churn
                    if len(filtered) != len(results):
                        summary["results"] = filtered
            except Exception:
                # Ignore malformed entries, keep going
                continue

        # Persist once
        self._persist()
        return removed

    # =================== Chunk Penalization Management ===================
    def get_penalized_chunk_ids(self) -> List[str]:
        """Return the list of penalized chunk ids (kept but deprioritized)."""
        return list(self.state.get("penalized_chunk_ids", []))

    def add_penalized_chunks(self, chunk_ids: List[str]) -> int:
        """Add chunk ids to the penalized list (idempotent). Returns number added."""
        if not chunk_ids:
            return 0
        existing = set(self.state.get("penalized_chunk_ids", []))
        before = len(existing)
        for cid in chunk_ids:
            if cid:
                existing.add(cid)
        self.state["penalized_chunk_ids"] = list(existing)
        self._persist()
        return len(existing) - before

    # =================== Typed Feedback Management ===================
    # region Typed Feedback Management

    def store_feedback_items_for_iteration(
        self, iteration: int, items: list[FeedbackStoredModel]
    ) -> None:
        """
        Persist reviewer-created items under the given iteration.
        Iteration is the container key; items are keyed by id inside it.
        """
        bucket = self.state["feedback_by_iteration"].setdefault(iteration, {})
        for it in items:
            bucket[it.id] = it
            self.state["item_index"][it.id] = iteration
        self._persist()

    def get_feedback_items_for_iteration(
        self, iteration: int, filter_by: Optional[FeedbackStatus] = None
    ) -> dict[str, list[FeedbackStoredModel]]:
        """
        Returns section -> [FeedbackStoredModel] for this iteration.
        If filter_by is provided, only items with status == filter_by are included.
        If filter_by is None, all items are included.
        """
        bucket = self.state["feedback_by_iteration"].get(iteration, {})
        out: dict[str, list[FeedbackStoredModel]] = {}
        for item in bucket.values():
            item_status = getattr(item, "status", None)
            # Normalize both to string for robust comparison
            if filter_by is None or (
                item_status is not None and str(item_status) == str(filter_by)
            ):
                out.setdefault(item.section, []).append(item)
        return out

    def get_feedback_item_by_id(self, item_id: str) -> Optional[FeedbackStoredModel]:
        it = self.state["item_index"].get(item_id)
        if it is None:
            return None
        return self.state["feedback_by_iteration"].get(it, {}).get(item_id)

    def update_feedback_item_status(self, item_id: str, status: str) -> bool:
        it = self.state["item_index"].get(item_id)
        if it is None:
            return False
        bucket = self.state["feedback_by_iteration"].get(it, {})
        item = bucket.get(item_id)
        if not item:
            return False
        item.status = status  # type: ignore
        self._persist()
        return True

    def set_feedback_item_comment(self, item_id: str, comment: str) -> bool:
        it = self.state["item_index"].get(item_id)
        if it is None:
            return False
        bucket = self.state["feedback_by_iteration"].get(it, {})
        item = bucket.get(item_id)
        if not item:
            return False
        item.writer_comment = comment  # type: ignore
        self._persist()
        return True

    def apply_feedback_updates(
        self, updates: list["WriterStatusUpdate"]
    ) -> dict[str, bool]:
        """
        Convenience: apply status + optional writer_comment for a batch of updates.
        Returns {feedback_id: True/False} for success per id.
        """
        results: dict[str, bool] = {}
        for upd in updates:
            ok = True
            try:
                self.update_feedback_item_status(upd.id, upd.status)
            except Exception:
                ok = False
            try:
                if upd.writer_comment is not None:
                    self.set_feedback_item_comment(upd.id, upd.writer_comment)
            except Exception:
                ok = False
            results[upd.id] = ok
        return results

    def replace_section_text(
        self,
        section: str,
        new_text: str,
        rebuild_and_persist: bool = True,
    ) -> "Article":
        """
        Replace the text of a single section. If section == '_overall', treat as full article replacement.
        Rebuilds full content from sections (when applicable) and persists.
        """
        article = self.get_current_draft_as_article()
        if not article:
            raise RuntimeError("No current draft to modify.")

        if section == "_overall":
            article.content = new_text
            # optional: do not try to sync sections here
        else:
            article.sections = dict(article.sections or {})
            article.sections[section] = new_text
            if rebuild_and_persist:
                article.content = build_full_article_content(
                    article.title, article.sections
                )

        if rebuild_and_persist:
            self.update_article_state(article)

        return article

    # endregion Typed Feedback Management

    # =================== Enhanced Draft Management ===================

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

    def update_article_state(self, article: Article) -> None:
        """
        Centralized method to update article state consistently,
        managing both historical archives and current state cache,
        while correctly handling partial updates.
        """
        iteration_key = str(self.state["iteration"])

        # --- Handle Draft Content (High priority change) ---
        new_draft_content = (
            article.content if article.content else self.state["current_draft"]
        )

        # --- Handle Sections (Dependent on new draft or explicit update) ---
        sections_to_use: Dict[str, str] = self.state["current_sections"].copy()

        if article.sections:
            # Case 1: Explicit sections provided -> use them
            sections_to_use.update(article.sections)
        elif article.content:
            # Case 2: New draft provided, but no explicit sections -> re-extract from draft
            sections_to_use = self._extract_sections_from_draft(article.content)
        # Case 3: Neither draft nor sections provided -> sections_to_use remains the existing self.state["current_sections"]

        # --- 1. Update History ---
        self.state["drafts_by_iteration"][iteration_key] = new_draft_content
        self.state["article_sections_by_iteration"][iteration_key] = sections_to_use

        # --- 2. Update Cache ---
        self.state["current_draft"] = new_draft_content
        self.state["current_sections"] = sections_to_use

        # --- 3. Update metadata if provided (standard update pattern) ---
        if article.metadata:
            self.state["metadata"].update(article.metadata)

        # --- 4. Persist changes ---
        self._persist()

    # =================== Suggested Queries Management ===================
    # region Suggested Queries Management

    def store_suggested_queries(self, iteration: int, queries: List[str]) -> None:
        """
        Store suggested queries from reviewer for a specific iteration.

        Args:
            iteration: Iteration number when queries were suggested
            queries: List of search query strings suggested by reviewer
        """
        if not queries:
            return

        self.state["suggested_queries_by_iteration"][iteration] = queries
        self._persist()
        logger.info(
            f"Stored {len(queries)} suggested queries for iteration {iteration}"
        )

    def get_suggested_queries(self, iteration: int) -> List[str]:
        """
        Retrieve suggested queries for a specific iteration.

        Args:
            iteration: Iteration number to retrieve queries for

        Returns:
            List of query strings, empty list if none exist
        """
        return self.state["suggested_queries_by_iteration"].get(iteration, [])

    def clear_suggested_queries(self, iteration: int) -> None:
        """
        Clear suggested queries for a specific iteration after they've been processed.

        Args:
            iteration: Iteration number to clear queries for
        """
        if iteration in self.state["suggested_queries_by_iteration"]:
            del self.state["suggested_queries_by_iteration"][iteration]
            self._persist()
            logger.info(f"Cleared suggested queries for iteration {iteration}")

    # endregion Suggested Queries Management
