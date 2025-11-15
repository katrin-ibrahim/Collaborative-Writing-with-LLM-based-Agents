# src/collaborative/agents/writer_v2.py
from datetime import datetime

import logging
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Literal, Optional, TypedDict

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    build_revision_batch_prompt,
    build_self_refine_prompt,
    build_single_section_revision_prompt,
    enhance_prompt_with_tom,
    outline_prompt,
    select_section_chunks_prompt,
    select_sections_chunks_batch_prompt,
    write_full_article_prompt,
    write_section_content_prompt,
    write_sections_batch_prompt,
)
from src.collaborative.memory.memory import SharedMemory
from src.collaborative.tom.theory_of_mind import AgentRole
from src.collaborative.utils.models import (
    ArticleOutlineValidationModel,
    BatchChunkSelectionModel,
    BatchSectionWritingModel,
    FeedbackStatus,
    FeedbackStoredModel,
    WriterValidationBatchModel,
    WriterValidationModel,
)
from src.collaborative.utils.writer_utils import (
    build_formatted_chunk_summaries,
    build_full_article_content,
)
from src.config.config_context import ConfigContext
from src.retrieval.factory import create_retrieval_manager
from src.utils.data.models import Article, Outline, ResearchChunk
from src.utils.link_extractor import LinkExtractor

logger = logging.getLogger(__name__)

retrieval_config = ConfigContext.get_retrieval_config()
MAX_QUERIES_CONFIG = (
    retrieval_config.num_queries
    if retrieval_config and hasattr(retrieval_config, "num_queries")
    else 5
)
MAX_CHUNKS_PER_SECTION = (
    retrieval_config.final_passages
    if retrieval_config and hasattr(retrieval_config, "final_passages")
    else 10
)


class ResearchGateDecision(BaseModel):
    action: Literal["continue", "retry"] = Field(
        description="continue to outline, or retry to do more research"
    )
    delete_chunk_ids: List[str] = Field(
        default_factory=list, description="Chunk IDs to delete"
    )
    new_queries: List[str] = Field(
        default_factory=list,
        description="New Wikipedia page titles to search (only if action=retry)",
    )
    reasoning: str = Field(default="", description="Brief explanation of decision")


# region Writer Validation Models
class QueryListValidationModel(BaseModel):
    """A list of search queries to execute, constrained by configuration."""

    queries: List[str] = Field(
        ...,
        # Constraints are now applied directly to the field
        min_length=1,
        max_length=MAX_QUERIES_CONFIG,
        description=f"A list of specific search queries (max {MAX_QUERIES_CONFIG}) based on the current context.",
    )


class EntityListValidationModel(BaseModel):
    """A list of entity names extracted from context."""

    entities: List[str] = Field(
        ...,
        min_length=0,
        max_length=10,
        description="A list of entity names (Wikipedia page titles) extracted from context.",
    )


class ChunkSelectionValidationModel(BaseModel):
    """The list of chunk IDs that are most relevant for writing the current section."""

    chunk_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_CHUNKS_PER_SECTION,
        description=f"A list of up to {MAX_CHUNKS_PER_SECTION} chunk_ids for the current section.",
    )

    @field_validator("chunk_ids", mode="before")
    @classmethod
    def coerce_chunk_ids_to_strings(cls, v: Any) -> List[str]:
        """Coerce chunk IDs to strings (LLMs often return them as integers)."""
        if not isinstance(v, list):
            return v
        return [str(item) for item in v]


class GateDecision(BaseModel):
    action: Literal["continue", "retry"]
    reasoning: Optional[str] = None


# ======================== MINIMAL STATE ========================


class MinimalState(TypedDict):
    """Minimal state for LangGraph compatibility."""


# ======================== WRITER AGENT ========================
class WriterV3(BaseAgent):
    """
    Simplified Writer agent with hybrid Python/LLM architecture.

    Workflow:
    - Iteration 0: research → create_outline() → write_all_sections()
    - Iteration 1+: get_pending_feedback → identify_target_sections →
                     revise_sections(feedback + chunks) → mark_feedback_status
    """

    def __init__(self):
        super().__init__()
        self.shared_memory: SharedMemory = ConfigContext.get_memory_instance()
        if not self.shared_memory:
            raise RuntimeError("SharedMemory instance not found in ConfigContext")

        # Build the LangGraph workflow
        self.graph = self._build_graph()
        self.config = ConfigContext.get_collaboration_config()
        self.retrieval_config = ConfigContext.get_retrieval_config()
        self.retrieval_manager = create_retrieval_manager(
            rm_type=self.retrieval_config.retrieval_manager
        )
        self.needs_refine = getattr(self.config, "should_self_refine", False)

        self._outline_ctx: Optional[str] = None
        self._research_retries: int = 0
        self._refine_retries: int = 0
        self._tried_queries: set = set()
        self._pending_gate_queries: Optional[List[str]] = None

        logger.info("Writer agent initialized with LangGraph architecture")

    def process(self) -> None:
        """Main public entry point - initializes state and runs graph."""
        self.iteration = self.shared_memory.get_iteration()
        logger.info(f"WriterV2 starting processing for iteration {self.iteration}")
        try:
            # LangGraph invocation only needs the starting state
            self.graph.invoke({})
            logger.info(f"Writer complete for iteration {self.iteration}")
        except Exception as e:
            logger.error(f"Writer workflow failed: {e}", exc_info=True)
            raise

    # ======================== GRAPH CONSTRUCTION ========================
    # region Graph Construction
    def _build_graph(self):
        """Build the LangGraph workflow with nodes and conditional edges."""
        workflow = StateGraph(MinimalState)

        # Nodes
        workflow.add_node("router", self._router_by_iteration)
        workflow.add_node("research", self._research)  # Initial Flow Only
        workflow.add_node("outline", self._outline)  # Initial Flow Only
        workflow.add_node("write", self._write_draft)  # Initial Flow Only
        workflow.add_node("revise", self._revise_draft)  # Revision Flow Only
        workflow.add_node("self_refine", self._self_refine)  # Both Flows (Optional)

        workflow.set_entry_point("router")

        # 1. Router Edges
        workflow.add_conditional_edges(
            "router",
            lambda state: self.flow_type,
            {
                "INITIAL_FLOW": "research",  # iter 0: research -> outline...
                "REVISION_FLOW": "revise",  # iter 1+: go directly to revise (no retrieval)
            },
        )

        # 2. Research -> Research Gate (pre-outline filtering)
        workflow.add_conditional_edges(
            "research",
            self._should_refine_research,
            {
                "retry": "research",
                "continue": "outline",
            },
        )

        # 4. Outline Refinement -> Write
        workflow.add_edge("outline", "write")

        # 5. Post-Content Edges (Shared for write and revise)
        workflow.add_conditional_edges(
            "write", self._should_self_refine, {"refine": "self_refine", "done": END}
        )
        workflow.add_edge("revise", END)

        return workflow.compile()

    # endregion Graph Construction

    # ======================== GRAPH NODES ========================

    # ======================= GRAPH CONDITIONALS =======================
    # region Graph Conditionals
    # ======================== ROUTER NODE ========================
    def _router_by_iteration(self, state: MinimalState) -> MinimalState:
        """Determine flow type based on iteration."""
        if self.iteration == 0:
            self.flow_type = "INITIAL_FLOW"
            logger.info("Router selected flow: INITIAL_FLOW")
        else:
            self.flow_type = "REVISION_FLOW"
            logger.info(
                f"Router selected flow: REVISION_FLOW for iteration {self.iteration}"
            )
        return state

    # ======================== RESEARCH GATE NODE ========================
    def _should_refine_research(self, state) -> str:
        """
        Pre-outline research gate: LLM filters irrelevant chunks and decides if more research needed.
        Can return "retry" (max 2 times) to do more research, or "continue" to outline.
        """
        chunk_summaries = self.shared_memory.get_search_summaries()
        topic = self.shared_memory.get_topic()

        if not chunk_summaries:
            logger.warning("No chunks to filter, proceeding to outline.")
            return "continue"

        total_chunks = sum(len(summary.results) for summary in chunk_summaries.values())

        if total_chunks == 0:
            logger.warning("No chunks available, proceeding to outline.")
            return "continue"

        # Track research attempts
        if not hasattr(self, "_research_retries"):
            self._research_retries = 0
            self._tried_queries = set()

        # Get queries we already tried
        tried_queries_str = (
            ", ".join(sorted(self._tried_queries))
            if self._tried_queries
            else "None yet"
        )

        logger.info(
            f"Pre-outline research gate (attempt {self._research_retries + 1}/3): LLM filtering {total_chunks} chunks"
        )
        logger.info(f"Already tried queries: {tried_queries_str}")

        # Build formatted chunk list for LLM
        from src.collaborative.utils.writer_utils import build_formatted_chunk_summaries

        chunks_formatted = build_formatted_chunk_summaries(
            chunk_summaries,
            max_content_pieces=None,
            fields=["description", "chunk_id"],
        )

        # Call LLM to filter and decide
        research_client = self.get_task_client("research")
        link_extractor = LinkExtractor()
        normalized_topic = topic.replace("_", " ")
        wiki_links = link_extractor.extract_topic_data(normalized_topic)

        try:
            from src.collaborative.agents.templates import build_research_gate_prompt

            prompt = build_research_gate_prompt(
                topic=topic,
                wiki_links=", ".join(wiki_links),
                chunks_formatted=chunks_formatted,
                tried_queries=tried_queries_str,
                retry_count=self._research_retries,
            )

            decision: ResearchGateDecision = research_client.call_structured_api(
                prompt=prompt,
                output_schema=ResearchGateDecision,
                system_prompt="You are a research gate. Decide if we need more research or can proceed to outline.",
                temperature=0.0,
            )

            logger.info(
                f"Research gate decision: {decision.action}, reasoning: {decision.reasoning}, new queries: {', '.join(decision.new_queries)}"
            )
            logger.debug(f"Chunks to delete: {len(decision.delete_chunk_ids)}")

        except Exception as e:
            logger.warning(f"LLM filtering failed: {e}, proceeding to outline")
            return "continue"

        # Delete bad chunks
        if decision.delete_chunk_ids:
            chunks_before = self.shared_memory.get_stored_chunks()
            logger.info(f"BEFORE pruning: {len(chunks_before)} chunks in memory")
            logger.debug(f"Deleting chunk IDs: {decision.delete_chunk_ids}")
            logger.debug(
                f"Deleting chunks with content excerpts: {[chunk.description for chunk in chunks_before if chunk.chunk_id in decision.delete_chunk_ids]}"
            )

            try:
                removed = self.shared_memory.delete_chunks_by_ids(
                    decision.delete_chunk_ids
                )
                logger.info(f"Pre-outline gate pruned {removed} irrelevant chunks")
            except Exception as del_exc:
                logger.warning(f"Failed to delete chunks: {del_exc}")

            chunks_after = self.shared_memory.get_stored_chunks()
            logger.info(f"AFTER pruning: {len(chunks_after)} chunks remaining")
            logger.debug(
                f"Remaining chunks with content excerpts: {[chunk.description for chunk in chunks_after]}"
            )

        # Check if we should retry with new queries
        if (
            decision.action == "retry"
            and self._research_retries < 2
            and decision.new_queries
        ):
            self._research_retries += 1
            logger.info(
                f"Research gate requesting retry with {len(decision.new_queries)} new queries"
            )

            # Store new queries for research node to use
            self._pending_gate_queries = decision.new_queries

            return "retry"

        if self._research_retries >= 2:
            logger.info("Max research retries (2) reached, proceeding to outline")

        return "continue"

    # ======================== SELF-REFINE GATE NODE ========================
    def _should_self_refine(self, state: MinimalState) -> str:
        """Conditional: Checks if self-refine is configured and needed (Post-content)."""
        if getattr(self.config, "should_self_refine", False) and self.needs_refine:
            logger.info("Self-refinement flag is set. Rerouting to self_refine.")
            return "refine"
        return "done"

    # endregion Graph Conditionals

    # ======================== INITIAL FLOW NODES ========================
    # region Initial Flow Nodes
    # ======================== RESEARCH NODE ==========================
    def _research(self, state: MinimalState) -> MinimalState:
        """Research node: two-phase research process (initial + secondary search)."""
        topic = self.shared_memory.get_topic()

        if _pending_gate_queries := getattr(self, "_pending_gate_queries", None):
            # Use queries proposed by research gate
            logger.info(
                f"Using {len(_pending_gate_queries)} queries from research gate"
            )
            self._execute_search_phase(_pending_gate_queries)
            self._tried_queries.update(_pending_gate_queries)
            # Clear pending queries after use
            self._pending_gate_queries = None
            return state

        link_extractor = LinkExtractor()
        # Normalize topic: replace underscores with spaces for Wikipedia lookup
        normalized_topic = topic.replace("_", " ")
        wiki_links = link_extractor.extract_topic_data(normalized_topic)

        # If no links found, use the topic itself as fallback
        if not wiki_links:
            logger.warning(
                f"No related links found for {normalized_topic}, using topic as single query"
            )
            wiki_links = [normalized_topic]

        logger.info(
            f"Starting research for topic: {topic} with {len(wiki_links)} queries"
        )
        self._execute_search_phase(wiki_links)
        self._tried_queries = set(wiki_links)

        return state

    # ======================== OUTLINE NODE ========================
    def _outline(self, state: MinimalState) -> MinimalState:
        """Outline node: create article outline using LLM reasoning with top chunk context."""
        topic = self.shared_memory.get_topic()

        # Log memory state at outline creation
        chunks_in_memory = self.shared_memory.get_stored_chunks()
        chunk_ids_in_memory = [c.chunk_id for c in chunks_in_memory]
        logger.info(
            f"OUTLINE CREATION: {len(chunk_ids_in_memory)} chunks available: {chunk_ids_in_memory}"
        )

        chunk_summaries = self.shared_memory.get_search_summaries()
        formatted_chunk_summaries = build_formatted_chunk_summaries(
            chunk_summaries,
            max_content_pieces=self.retrieval_config.final_passages,
            fields=["description"],
        )

        outline_prompt_str = outline_prompt(topic, formatted_chunk_summaries)

        tom_context = self._get_tom_context(
            action="create_outline",
            topic=topic,
        )
        try:
            outline_client = self.get_task_client("create_outline")
            final_prompt = enhance_prompt_with_tom(outline_prompt_str, tom_context)
            outline_model = outline_client.call_structured_api(
                prompt=final_prompt, output_schema=ArticleOutlineValidationModel
            )
            outline = Outline(title=topic, headings=outline_model.headings)
        except Exception as e:
            logger.warning(f"LLM outline creation failed: {e}, using generic fallback")
            generic_headings = [
                "Introduction and Background",
                "Key Details and Context",
                "Main Events and Analysis",
                "Important Outcomes",
                "Impact and Significance",
                "Conclusion and Summary",
            ]
            outline = Outline(title=topic, headings=generic_headings)
            logger.warning("Fallback outline created.")

        # Store outline in memory (whether from LLM or fallback)
        self.initial_outline = outline
        self.shared_memory.state["initial_outline"] = outline
        logger.info(
            f"Created outline with {len(outline.headings)} sections: {outline.headings}"
        )
        return state

    # ======================== WRITE NODE ========================
    def _write_draft(self, state: MinimalState) -> MinimalState:
        """Write draft node."""
        topic: str = self.shared_memory.state.get("topic", "")
        outline: Optional["Outline"] = self.shared_memory.get_outline()
        if not outline or not outline.headings:
            raise RuntimeError("Write node: outline is missing or empty.")

        tom_context = self._get_tom_context(
            action="content_write",
            topic=topic,
        )
        chunk_summaries = self.shared_memory.get_search_summaries()
        chunk_summaries_str = build_formatted_chunk_summaries(
            chunk_summaries,
            max_content_pieces=self.retrieval_config.final_passages,
            fields=["description", "chunk_id"],
        )
        mode = getattr(getattr(self, "config", None), "writing_mode", "section")
        if mode not in ("section", "full_article"):
            mode = "section"

        if mode == "section":
            article = self._write_sections(
                topic=topic,
                outline=outline,
                chunk_summaries_str=chunk_summaries_str,
                tom_context=tom_context,
            )
        else:
            article = self._write_full_article(
                topic=topic,
                outline=outline,
                chunk_summaries_str=chunk_summaries_str,
                tom_context=tom_context,
            )

        self.shared_memory.update_article_state(article)
        return state

    # ======================== SELF-REFINE NODE ========================
    def _self_refine(self, state: MinimalState) -> MinimalState:
        """Self-refine node: refine the full article content."""
        article = self.shared_memory.get_current_draft_as_article()
        if not article or not (article.content or "").strip():
            return state  # nothing to polish

        client = self.get_task_client("self_refine")
        prompt = build_self_refine_prompt(
            title=article.title,
            current_text=article.content,
        )

        try:
            refined = (client.call_api(prompt=prompt) or "").strip()
            if refined:
                # Replace ONLY the full-text body; do not touch sections for speed.
                article.content = refined
                self.shared_memory.update_article_state(article)
        except Exception as e:
            logger.warning(f"[self_refine] failed: {e}")

        return state

    # endregion Initial Flow Nodes

    # ======================== REVISION FLOW NODES ========================
    # region Revision Flow Nodes
    def _revise_draft(self, state: MinimalState) -> MinimalState:
        """
        Graph node: revision step.
        - Check if reviewer suggested new queries and execute searches
        - Collect sections that have PENDING feedback items (current iteration).
        - One LLM call returns WriterValidationBatchModel with per-section updates.
        - Apply status updates + content rewrites.
        """
        logger.info("[REVISE NODE] Starting _revise_draft")
        iteration = self.shared_memory.state.get("iteration", 0)
        logger.info(f"[REVISE NODE] Current iteration: {iteration}")

        # Execute reviewer-suggested queries if available
        reviewer_queries = self.shared_memory.state.get("reviewer_suggested_queries")
        if reviewer_queries:
            logger.info(
                f"[REVISE NODE] Found {len(reviewer_queries)} reviewer-suggested queries"
            )
            self._execute_search_phase(reviewer_queries)
            # Clear the queries after execution
            self.shared_memory.state["reviewer_suggested_queries"] = None
            logger.info(
                "[REVISE NODE] Executed reviewer queries and cleared from memory"
            )

        article = self.shared_memory.get_current_draft_as_article()
        if not article:
            logger.error("[REVISE NODE] No current draft article found!")
            raise RuntimeError("Revise node: no current draft in memory.")

        # collect sections -> [pending items]
        # Feedback is stored under the previous iteration (when reviewer generated it)
        feedback_iteration = iteration - 1 if iteration > 0 else 0
        logger.info(
            f"[REVISE NODE] Getting feedback items from iteration {feedback_iteration} (current={iteration})"
        )
        pending_by_section = self.shared_memory.get_feedback_items_for_iteration(
            feedback_iteration, FeedbackStatus.PENDING
        )
        logger.info(
            f"[REVISE NODE] Found pending feedback for {len(pending_by_section)} sections"
        )
        if not pending_by_section:
            logger.info("[REVISE NODE] No pending feedback - returning early")
            return state  # nothing to do

        mode = getattr(getattr(self, "config", None), "revise_mode", "pending")
        logger.info(f"[REVISE NODE] Revise mode: {mode}")
        if mode == "section":
            # Sequential: one LLM call per section (SLOW but more focused)
            logger.info(
                f"[REVISE NODE] Using SEQUENTIAL mode ({len(pending_by_section)} LLM calls, one per section)"
            )
            self._revise_sections_sequential(article, pending_by_section)
        else:  # "pending" or any other value - default to batch/pending mode
            # Batch/Pending: one LLM call for all pending sections (FAST, recommended)
            logger.info(
                f"[REVISE NODE] Using BATCH/PENDING mode (1 LLM call for all {len(pending_by_section)} sections)"
            )
            self._revise_sections_batch(article, pending_by_section)

        # persist
        self.shared_memory.update_article_state(article)
        return state

    # endregion Revision Flow Nodes

    # ======================== WRITER V2 HELPERS ========================
    # region WriterV2 Helpers

    def _get_tom_context(self, action: str, topic: str) -> Optional[str]:
        """Generate Theory of Mind context if available."""
        if not (
            hasattr(self.shared_memory, "tom_module")
            and self.shared_memory.tom_module
            and self.shared_memory.tom_module.enabled
        ):
            return None

        try:
            from src.collaborative.agents.templates import (
                build_directive_tom_context_for_writer,
            )

            # Writer predicts how Reviewer will respond
            tom_prediction = self.shared_memory.tom_module.predict_agent_response(
                predictor=AgentRole.WRITER,
                target=AgentRole.REVIEWER,
                context={
                    "topic": topic,
                    "action": action,
                },
            )

            return build_directive_tom_context_for_writer(
                predicted_action=tom_prediction.predicted_action,
                confidence=tom_prediction.confidence,
                reasoning=tom_prediction.reasoning,
            )
        except Exception as e:
            logger.warning(f"ToM prediction failed: {e}")
            return None

    def _execute_search_phase(
        self, queries: List[str], rm_type: Optional[str] = None
    ) -> None:
        """
        Executes one or more search queries concurrently using the Retrieval Manager
        and stores the globally processed results in shared memory (D.R.Y. logic).
        """
        logger.info(f"Preparing search execution for {len(queries)} queries.")

        if not queries:
            logger.warning(
                "Attempted to execute search phase with an empty query list."
            )
            return

        # 1. Get Configuration and Manager Instance
        rm_type = self.retrieval_config.retrieval_manager
        try:
            retrieval_manager = create_retrieval_manager(rm_type)
        except NotImplementedError:
            logger.error(f"Cannot initialize retrieval manager of type: {rm_type}")
            return

        # 2. Define the unique search identifier (the key for memory storage)
        # This is the "mega-query" key used for storage consistency
        search_id = " | ".join(queries)

        try:
            logger.info(
                f"Executing concurrent search for {len(queries)} queries. ID: {search_id}"
            )

            # A. Call the RM's concurrent method (RM handles caching, dedup, and re-ranking)
            # The 'topic' argument helps the RM contextually score relevance across the batch.
            topic = self.shared_memory.get_topic()
            if not topic:
                logger.warning(
                    "No topic found in shared memory; proceeding with empty topic context."
                )
                topic = ""
            if rm_type == "wiki":
                all_results: List[ResearchChunk] = retrieval_manager.search_concurrent(
                    query_list=queries, topic=topic
                )
                logger.debug(
                    f"Retrieved {len(all_results)} results from wiki retrieval manager."
                )
                logger.debug(
                    f"debug excerpts: {[chunk.description for chunk in all_results]}"
                )
            else:
                # faiss has segfault issues with concurrent calls, so we use sequential for now
                all_results: List[ResearchChunk] = retrieval_manager.search(
                    query_list=queries, topic=topic
                )

            # This utility handles calling shared_memory.store_research_chunks()
            # and shared_memory.store_search_summary()
            self._process_and_store_chunks(
                search_id=search_id,  # The key for memory storage
                source_queries=queries,  # The original list of input queries
                chunks=all_results,
                rm_type=rm_type,
            )

            # Log current memory state after storing
            chunks_in_memory = self.shared_memory.get_stored_chunks()
            chunk_ids_in_memory = [c.chunk_id for c in chunks_in_memory]
            logger.info(
                f"AFTER storing: Total {len(chunk_ids_in_memory)} chunks in memory: {chunk_ids_in_memory}"
            )

            # Track queries in shared memory to prevent redundant searches
            if "all_searched_queries" not in self.shared_memory.state:
                self.shared_memory.state["all_searched_queries"] = set()
            for query in queries:
                self.shared_memory.state["all_searched_queries"].add(query)
            logger.debug(
                f"Tracked {len(queries)} queries. Total tracked: {len(self.shared_memory.state['all_searched_queries'])}"
            )

        except Exception as exc:
            logger.error(
                f"Critical search failure for search_id '{search_id}': {exc}",
                exc_info=True,
            )
            # Store a structured failed search summary for tracking
            self.shared_memory.store_search_summary(
                search_id,
                {
                    "success": False,
                    "message": f"Critical search failure: {exc}",
                    "source_queries": queries,
                    "rm_type": rm_type,
                },
            )

    def _process_and_store_chunks(
        self,
        search_id: str,
        source_queries: List[str],
        chunks: List[ResearchChunk],
        rm_type: str,
    ) -> Dict[str, Any]:
        """
        Processes a list of ResearchChunks retrieved by the RM (potentially from a concurrent search),
        stores them in shared memory, and returns the structured search summary.

        This function is the D.R.Y. implementation that replaces the old search_and_retrieve tool logic.
        """
        if not chunks:
            # Create a summary for a successful search that returned no results
            search_result = {
                "source_queries": source_queries,
                "rm_type": rm_type,
                "total_chunks": 0,
                "results": [],  # Using 'results' key to match SearchSummary Pydantic model
                "metadata": {"message": f"No chunks found for search_id: {search_id}"},
                "success": True,
            }
            self.shared_memory.store_search_summary(search_id, search_result)
            return search_result

        # The RM has already handled chunk creation, deduplication, and ranking.
        # 1. Store chunks and get the required summaries
        # NOTE: The store_research_chunks method returns the list of summary objects/dicts needed for SearchSummary.results
        chunk_summaries = self.shared_memory.store_chunks(chunks)

        # 2. Build the final search result summary dictionary
        search_result = {
            "source_queries": source_queries,
            "rm_type": rm_type,
            "total_chunks": len(chunk_summaries),
            "results": chunk_summaries,
            "metadata": {
                "message": f"Found and stored {len(chunk_summaries)} unique chunks.",
                "source_rm": rm_type,
            },
            "success": True,
        }

        # 3. Store the search summary in memory using the unique search_id
        self.shared_memory.store_search_summary(search_id, search_result)

        return search_result

    def _write_sections(
        self,
        topic: str,
        outline: "Outline",
        chunk_summaries_str: str,
        tom_context: Optional[str],
    ) -> "Article":
        """Batch section writing with batch size 2 for better quality."""
        BATCH_SIZE = 1
        sections_out: Dict[str, str] = {}
        headings = outline.headings

        # Step 1: Batched chunk selection (split into batches of 10 to avoid truncation)
        SELECTION_BATCH_SIZE = 5
        section_chunks_map = {}
        selection_client = self.get_task_client("research")

        logger.info(
            f"Batch selecting chunks for {len(headings)} sections in batches of {SELECTION_BATCH_SIZE}"
        )

        for batch_start in range(0, len(headings), SELECTION_BATCH_SIZE):
            batch_end = min(batch_start + SELECTION_BATCH_SIZE, len(headings))
            batch_headings = headings[batch_start:batch_end]

            try:
                select_prompt = select_sections_chunks_batch_prompt(
                    batch_headings,
                    topic,
                    chunk_summaries_str,
                    self.retrieval_config.final_passages,
                )

                batch_selection = selection_client.call_structured_api(
                    prompt=select_prompt, output_schema=BatchChunkSelectionModel
                )

                for selection in batch_selection.selections:
                    section_chunks_map[selection.section_heading] = selection.chunk_ids

                logger.info(
                    f"  Batch {batch_start//SELECTION_BATCH_SIZE + 1}: "
                    f"Selected chunks for sections {batch_start+1}-{batch_end}"
                )
            except Exception as e:
                logger.warning(
                    f"Batch chunk selection failed for sections {batch_start+1}-{batch_end}: {e}, "
                    f"falling back to sequential selection"
                )
                # Fallback: sequential chunk selection for this batch
                for heading in batch_headings:
                    try:
                        select_prompt = select_section_chunks_prompt(
                            heading,
                            topic,
                            chunk_summaries_str,
                            self.retrieval_config.final_passages,
                        )
                        sel = selection_client.call_structured_api(
                            prompt=select_prompt,
                            output_schema=ChunkSelectionValidationModel,
                        )
                        section_chunks_map[heading] = getattr(sel, "chunk_ids", [])
                    except Exception:
                        section_chunks_map[heading] = []

        total_selected = sum(len(chunks) for chunks in section_chunks_map.values())
        avg_per_section = (
            total_selected / len(section_chunks_map) if section_chunks_map else 0
        )
        logger.info(
            f"Chunk selection completed: {len(section_chunks_map)} sections, "
            f"{total_selected} total chunks ({avg_per_section:.1f} avg per section)"
        )

        if BATCH_SIZE > 1:
            # Step 2: Write sections in batches
            for batch_start in range(0, len(headings), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(headings))
                batch_headings = headings[batch_start:batch_end]

                logger.info(
                    f"Writing batch {batch_start//BATCH_SIZE + 1}: sections {batch_start+1}-{batch_end} of {len(headings)}"
                )

                sections_with_chunks = []
                for heading in batch_headings:
                    chunk_ids = section_chunks_map.get(heading, [])
                    if not chunk_ids:
                        sections_with_chunks.append(
                            {"section_heading": heading, "relevant_info": ""}
                        )
                        continue
                    chunk_models = self.shared_memory.get_chunks_by_ids(chunk_ids)
                    if chunk_models:
                        chunks_str = ", ".join(
                            f"{cid} {{content: {m.content}, score: {m.metadata.get('relevance_score')}, url: {m.url or 'N/A'}}}"
                            for cid, m in chunk_models.items()
                        )
                        sections_with_chunks.append(
                            {"section_heading": heading, "relevant_info": chunks_str}
                        )
                    else:
                        sections_with_chunks.append(
                            {"section_heading": heading, "relevant_info": ""}
                        )

                if not sections_with_chunks:
                    continue

                write_prompt = write_sections_batch_prompt(sections_with_chunks, topic)
                final_prompt = enhance_prompt_with_tom(write_prompt, tom_context)
                writer_client = self.get_task_client("writer")
                try:
                    batch_result = writer_client.call_structured_api(
                        prompt=final_prompt, output_schema=BatchSectionWritingModel
                    )
                    for section_model in batch_result.sections:
                        section_heading = section_model.section_heading
                        content = section_model.content
                        if content and content.strip():
                            sections_out[section_heading] = content.strip()
                except Exception as e:
                    logger.warning(
                        f"Batch section writing failed: {e}, falling back to sequential"
                    )
                    for item in sections_with_chunks:
                        heading = item["section_heading"]
                        relevant_info = item["relevant_info"]
                        write_prompt = write_section_content_prompt(
                            heading, topic, relevant_info
                        )
                        final_prompt = enhance_prompt_with_tom(
                            write_prompt, tom_context
                        )
                        try:
                            text = (
                                writer_client.call_api(prompt=final_prompt) or ""
                            ).strip()
                            if text:
                                sections_out[heading] = text
                        except Exception:
                            continue
        else:
            # Step 2: Sequential section writing
            for heading in headings:
                logger.info(f"Writing section: {heading}")
                chunk_ids = section_chunks_map.get(heading, [])
                chunk_models = self.shared_memory.get_chunks_by_ids(chunk_ids)

                if not chunk_models:
                    # Still attempt to write using heading alone for minimal content
                    text = self._write_single_section(
                        section=heading,
                        topic=topic,
                        chunks_str="",
                        tom_context=tom_context,
                    )
                    if text:
                        sections_out[heading] = text
                    continue
                chunks_str = ", ".join(
                    f"{cid} {{content: {m.content}, score: {m.metadata.get('relevance_score')}, url: {m.url or 'N/A'}}}"
                    for cid, m in chunk_models.items()
                )
                text = self._write_single_section(
                    section=heading,
                    topic=topic,
                    chunks_str=chunks_str,
                    tom_context=tom_context,
                )
                if text:
                    sections_out[heading] = text

        article = Article(
            title=topic,
            content=build_full_article_content(topic, sections_out),
            sections=sections_out,
            metadata={
                "iteration": self.shared_memory.state.get("iteration", 0),
                "sections_count": len(sections_out),
                "writing_mode": "section_batch",
                "batch_size": BATCH_SIZE,
            },
        )
        return article

    def _write_full_article(
        self,
        topic: str,
        outline: "Outline",
        chunk_summaries_str: str,
        tom_context: Optional[str],
    ) -> "Article":
        """One-shot full article generation."""
        base_prompt = write_full_article_prompt(
            topic=topic,
            headings=outline.headings,
            relevant_info=chunk_summaries_str,
        )
        prompt = enhance_prompt_with_tom(base_prompt, tom_context)

        client = self.get_task_client("writer")
        text = (client.call_api(prompt=prompt) or "").strip()

        sections = self._parse_markdown_sections(text)

        return Article(
            title=topic,
            content=text,
            sections=sections,
            metadata={
                "iteration": self.shared_memory.state.get("iteration", 0),
                "sections_count": len(sections),
                "writing_mode": "full_article",
                "timestamp": datetime.now().isoformat(),
            },
        )

    def _parse_markdown_sections(self, content: str) -> Dict[str, str]:
        """
        Parse markdown content into sections based on H2 headers (##).
        Returns a dictionary mapping section headings to their content.
        """
        sections = {}
        lines = content.split("\n")
        current_heading = None
        current_content = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                if current_heading and current_content:
                    sections[current_heading] = "\n".join(current_content).strip()
                current_heading = stripped[3:].strip()
                current_content = []
            elif current_heading is not None:
                current_content.append(line)

        if current_heading and current_content:
            sections[current_heading] = "\n".join(current_content).strip()

        return sections

    def _write_single_section(
        self,
        section: str,
        topic: str,
        chunks_str: str,
        tom_context: Optional[str] = None,
    ) -> Optional[str]:
        """Single section writer (structured selection + write)."""
        try:

            write_prompt = write_section_content_prompt(section, topic, chunks_str)
            final_prompt = enhance_prompt_with_tom(write_prompt, tom_context)

            client = self.get_task_client("writer")
            return (client.call_api(prompt=final_prompt) or "").strip()
        except Exception:
            return None

    def _revise_sections_batch(
        self,
        article: "Article",
        pending_by_section: dict[str, list["FeedbackStoredModel"]],
    ) -> None:
        """
        Batch mode: one LLM call to revise all sections that have PENDING feedback.
        """
        research_summary = self.shared_memory.get_search_summaries()
        research_ctx = build_formatted_chunk_summaries(
            research_summary,
            max_content_pieces=self.retrieval_config.final_passages,
            fields=["description"],
        )
        # 1) Prompt → batch model
        batch_prompt = build_revision_batch_prompt(
            article, pending_by_section, research_ctx
        )
        client = self.get_task_client("revision")
        batch_model = client.call_structured_api(
            prompt=batch_prompt, output_schema=WriterValidationBatchModel
        )

        # 2) Map feedback id -> section for routing patches
        id_to_section = {
            it.id: sec for sec, items in pending_by_section.items() for it in items
        }

        # 3) Apply feedback updates and collect section patches
        section_patches: dict[str, str] = {}

        for vm in batch_model.items:
            # feedback updates (via memory)
            # Log what statuses the writer is setting
            status_counts = {}
            for upd in vm.updates:
                status_val = (
                    upd.status if isinstance(upd.status, str) else upd.status.value
                )
                status_counts[status_val] = status_counts.get(status_val, 0) + 1
            logger.info(f"[revise/batch] Writer setting statuses: {status_counts}")

            results = self.shared_memory.apply_feedback_updates(vm.updates)

            # infer target section from first resolvable id; fallback to '_overall'
            target_section = next(
                (
                    id_to_section.get(upd.id)
                    for upd in vm.updates
                    if id_to_section.get(upd.id) is not None
                ),
                "_overall",
            )
            if not isinstance(target_section, str):
                target_section = "_overall"

            # warn for any prompted ids in that section not updated
            prompted_ids = {it.id for it in pending_by_section.get(target_section, [])}
            missing = [fid for fid in prompted_ids if not results.get(fid, False)]
            if missing:
                logger.warning(
                    f"[revise/batch] Section '{target_section}': missing/failed updates for ids={sorted(missing)}"
                )

            # we only support partial section rewrites—coerce if needed
            if vm.content_type != "partial_section":
                logger.warning(
                    f"[revise/batch] Model returned '{vm.content_type}', coercing to 'partial_section'"
                )

            section_patches[target_section] = vm.updated_content

        # 4) Apply patches → rebuild → persist once
        article.sections = dict(article.sections or {})
        overall_content = None

        for sec, text in section_patches.items():
            if sec == "_overall":
                overall_content = text
            else:
                article.sections[sec] = text

        if overall_content:
            article.content = overall_content
        elif article.sections:
            try:
                article.content = build_full_article_content(
                    article.title, article.sections
                )
            except Exception as e:
                logger.warning(f"[revise/batch] Rebuild failed: {e}")

        self.shared_memory.update_article_state(article)

    def _revise_sections_sequential(
        self,
        article: "Article",
        pending_by_section: dict[str, list["FeedbackStoredModel"]],
    ) -> None:
        """
        Sequential mode: one LLM call per section that has PENDING feedback.
        """
        client = self.get_task_client("revision")
        research_summary = self.shared_memory.get_search_summaries()
        research_ctx = build_formatted_chunk_summaries(
            research_summary,
            max_content_pieces=self.retrieval_config.final_passages,
            fields=["description"],
        )

        for section, items in pending_by_section.items():
            # 1) Prompt → single section model
            prompt = build_single_section_revision_prompt(
                article, section, items, research_ctx
            )
            vm = client.call_structured_api(
                prompt=prompt, output_schema=WriterValidationModel
            )

            # Log what statuses the writer is setting
            status_counts = {}
            for upd in vm.updates:
                status_val = (
                    upd.status if isinstance(upd.status, str) else upd.status.value
                )
                status_counts[status_val] = status_counts.get(status_val, 0) + 1
            logger.info(
                f"[revise/sequential] Section '{section}' writer setting statuses: {status_counts}"
            )

            # 2) feedback updates (via memory)
            results = self.shared_memory.apply_feedback_updates(vm.updates)
            prompted_ids = {it.id for it in items}
            missing = [fid for fid in prompted_ids if not results.get(fid, False)]
            if missing:
                logger.warning(
                    f"[revise/single] Section '{section}': missing/failed updates for ids={sorted(missing)}"
                )

            # 3) apply section patch (coerce to partial)
            if vm.content_type != "partial_section":
                logger.warning(
                    f"[revise/single] Model returned '{vm.content_type}', coercing to 'partial_section'"
                )

            if section == "_overall":
                article.content = vm.updated_content
            else:
                article.sections = dict(article.sections or {})
                article.sections[section] = vm.updated_content

        # 4) Rebuild from sections if we have them and didn't just do _overall
        if article.sections and not any(
            section == "_overall" for section in pending_by_section.keys()
        ):
            try:
                article.content = build_full_article_content(
                    article.title, article.sections
                )
            except Exception as e:
                logger.warning(f"[revise/single] Rebuild failed: {e}")

        self.shared_memory.update_article_state(article)


# endregion WriterV2 Helpers
