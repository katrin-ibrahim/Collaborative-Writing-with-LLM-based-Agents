# src/collaborative/agents/writer_v2.py
import logging
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, TypedDict

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    build_outline_gate_prompt,
    build_research_gate_prompt,
    build_revision_batch_prompt,
    build_self_refine_prompt,
    build_single_section_revision_prompt,
    enhance_prompt_with_tom,
    outline_prompt,
    refine_outline_prompt,
    search_query_generation_prompt,
    select_section_chunks_prompt,
    write_full_article_prompt,
    write_section_content_prompt,
)
from src.collaborative.memory.memory import SharedMemory
from src.collaborative.tom.theory_of_mind import AgentRole
from src.collaborative.utils.models import (
    ArticleOutlineValidationModel,
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


class ChunkSelectionValidationModel(BaseModel):
    """The list of chunk IDs that are most relevant for writing the current section."""

    chunk_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_CHUNKS_PER_SECTION,
        description=f"A list of up to {MAX_CHUNKS_PER_SECTION} chunk_ids for the current section.",
    )


class GateDecision(BaseModel):
    action: Literal["continue", "retry"]
    reasoning: Optional[str] = None


class ResearchSteerModel(BaseModel):
    """Structured output for the research gate.

    Either recommend chunk deletions and/or new queries. The action controls routing.
    """

    action: Literal["continue", "retry"]
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief rationale explaining deletions and/or new queries",
    )
    delete_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Chunk ids to delete due to redundancy, low relevance, or low quality",
    )
    new_queries: List[str] = Field(
        default_factory=list,
        min_length=0,
        max_length=MAX_QUERIES_CONFIG,
        description="Additional targeted queries to improve coverage",
    )


# ======================== MINIMAL STATE ========================


class MinimalState(TypedDict):
    """Minimal state for LangGraph compatibility."""


# ======================== WRITER AGENT ========================
class WriterV2(BaseAgent):
    """
    Simplified Writer agent with hybrid Python/LLM architecture.

    Workflow:
    - Iteration 0: direct_search(topic) → generate_queries(top_chunk) →
                   search_queries → create_outline(top_chunk) → write_all_sections(chunks)
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

        self._research_ctx: Optional[List[str]] = None
        self._outline_ctx: Optional[str] = None
        self._research_retries: int = 0
        self._refine_retries: int = 0

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
        workflow.add_node(
            "refine_research", self._refine_research
        )  # Initial Flow Only - Gate after outline
        workflow.add_node("refine_outline", self._refine_outline)  # Initial Flow Only
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
                "REVISION_FLOW": "revise",  # iter 1+: revise
            },
        )

        # 2. Research -> Outline (no gate, just collect chunks)
        workflow.add_edge("research", "outline")

        # 3. Outline -> Research Gate (with outline context)
        workflow.add_conditional_edges(
            "outline",
            self._should_refine_research,  # LLM gate with outline context
            {
                "retry": "refine_research",
                "continue_outline": "refine_outline",
                "continue_write": "write",
            },
        )

        # 4. Research Gate -> back to Research or continue
        workflow.add_conditional_edges(
            "refine_research",
            lambda state: "research" if self._research_ctx else "refine_outline",
            {"research": "research", "refine_outline": "refine_outline"},
        )

        # 5. Outline Refinement
        workflow.add_conditional_edges(
            "refine_outline",
            self._should_refine_outline,
            {"retry": "refine_outline", "continue": "write"},
        )

        # 6. Post-Content Edges (Shared for write and revise)
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
        Decide: do another research pass ('retry') or move on to outline refinement or writing.
        Uses outline context to make informed decisions about chunk relevance and gaps.
        Reviews all chunks via batched LLM calls to ensure complete coverage.
        """
        if self._research_retries >= 2:
            logger.info(
                "Max research retries reached, proceeding to outline refinement."
            )
            return "continue_outline"

        chunk_summaries = self.shared_memory.get_search_summaries()
        topic = self.shared_memory.get_topic()
        outline = self.shared_memory.get_outline()

        if not outline or not outline.headings:
            logger.warning(
                "No outline available for research gate, skipping refinement."
            )
            return "continue_write"

        outline_str = "\n".join(outline.headings)
        writer_client = self.get_task_client("writer")

        batch_size = (
            self.retrieval_config.final_passages if self.retrieval_config else 10
        )

        all_delete_ids = []
        all_new_queries = []
        all_decisions = []

        total_chunks = sum(len(summary.results) for summary in chunk_summaries.values())
        total_batches = (total_chunks + batch_size - 1) // batch_size

        logger.info(
            f"Research gate (post-outline) reviewing {total_chunks} chunks in {total_batches} batch(es)"
        )

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_chunks)

            chunks_formatted = build_formatted_chunk_summaries(
                chunk_summaries,
                max_content_pieces=None,
                fields=["description", "chunk_id"],
                start_idx=start_idx,
                end_idx=end_idx,
            )

            try:
                prompt = build_research_gate_prompt(
                    topic,
                    outline_str,
                    batch_idx,
                    total_batches,
                    chunks_formatted,
                    MAX_QUERIES_CONFIG,
                )

                dec: ResearchSteerModel = writer_client.call_structured_api(
                    prompt=prompt,
                    output_schema=ResearchSteerModel,
                    system_prompt="Answer with JSON only.",
                    temperature=0.0,
                )

                all_decisions.append(dec.action)
                if getattr(dec, "delete_chunk_ids", None):
                    all_delete_ids.extend(dec.delete_chunk_ids)
                if getattr(dec, "new_queries", None):
                    all_new_queries.extend(dec.new_queries)

            except Exception as e:
                logger.warning(f"Batch {batch_idx + 1} review failed: {e}")
                continue

        all_delete_ids = list(dict.fromkeys(all_delete_ids))
        all_new_queries = list(dict.fromkeys(all_new_queries))[:MAX_QUERIES_CONFIG]

        if all_delete_ids:
            try:
                removed = self.shared_memory.delete_chunks_by_ids(all_delete_ids)
                logger.info(f"Research gate deleted {removed} chunks: {all_delete_ids}")
            except Exception as del_exc:
                logger.warning(f"Failed to delete chunks {all_delete_ids}: {del_exc}")

        if all_new_queries:
            self._research_ctx = all_new_queries
            logger.info(f"Research gate proposed {len(all_new_queries)} new queries")

        should_retry = "retry" in all_decisions and all_new_queries

        if should_retry:
            self._research_retries += 1
            return "retry"

        return "continue_outline" if self._refine_retries < 1 else "continue_write"

    # ======================== OUTLINE GATE NODE ========================
    def _should_refine_outline(self, state) -> str:
        """
        Decide: refine outline once ('retry') or proceed to writing ('continue').
        """
        if self._refine_retries >= 1:
            logger.info(
                "Max outline refinement retries reached, proceeding to writing."
            )
            return "continue"

        topic = self.shared_memory.get_topic()
        outline = self.shared_memory.get_outline()
        outline_str = (
            "\n".join(outline.headings) if outline else "No outline available."
        )
        writer_client = self.get_task_client("writer")
        self._refine_retries += 1
        try:
            dec: GateDecision = writer_client.call_structured_api(
                prompt=build_outline_gate_prompt(topic, outline_str, self._outline_ctx),
                output_schema=GateDecision,
                system_prompt="Answer with JSON only.",
                temperature=0.0,
                max_tokens=120,
            )
            if dec.action == "retry":
                self._outline_ctx = dec.reasoning
                self._refine_retries += 1
                return "retry"
            else:
                return "continue"
        except Exception:
            return "continue"  # fail safe

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
        logger.info(f"Starting research for topic: {topic}")

        # If we already have validated queries in context (from the gate), execute them
        if self._research_ctx and len(self._research_ctx) > 0:
            # Deduplicate and cap to configured maximum
            seen = set()
            deduped = []
            for q in self._research_ctx:
                if q and q not in seen:
                    seen.add(q)
                    deduped.append(q)
            max_q = (
                self.retrieval_config.num_queries
                if self.retrieval_config
                and hasattr(self.retrieval_config, "num_queries")
                else MAX_QUERIES_CONFIG
            )
            queries = deduped[:max_q]
            if queries:
                logger.info(f"Research: executing {len(queries)} gate-provided queries")
                self._execute_search_phase(queries=queries)
            else:
                logger.info(
                    "Research: research_ctx had no usable queries after dedup/cap"
                )
            # Clear after consumption to avoid reuse
            self._research_ctx = None
            return state

        # No context provided -> first pass: initial topic search + generated queries
        logger.info(
            "Research: no existing context; running initial topic search + generated queries"
        )
        self._execute_search_phase(queries=[topic], rm_type="faiss")

        top_chunk = self._get_best_chunk()
        if not top_chunk:
            logger.warning(
                "Initial search provided no context. Proceeding without secondary queries."
            )
            return state

        secondary_queries = self._generate_queries(topic=topic, top_chunk=top_chunk)
        if secondary_queries:
            self._execute_search_phase(queries=secondary_queries)
        else:
            logger.info("No secondary queries generated by LLM. Search complete.")
        return state

    # ======================== REFINE RESEARCH NODE ========================
    def _refine_research(self, state: MinimalState) -> MinimalState:
        """Execute additional research queries suggested by the research gate."""
        if not self._research_ctx or len(self._research_ctx) == 0:
            logger.info("No new queries from research gate, proceeding.")
            return state

        logger.info(
            f"Refine research: executing {len(self._research_ctx)} queries from gate"
        )
        self._execute_search_phase(queries=self._research_ctx)
        self._research_ctx = None
        return state

    # ======================== OUTLINE NODE ========================
    def _outline(self, state: MinimalState) -> MinimalState:
        """Outline node: create article outline using LLM reasoning with top chunk context."""
        topic = self.shared_memory.get_topic()
        top_chunk = self._get_best_chunk()
        top_chunk_text = top_chunk.content if top_chunk else ""
        chunk_summaries = self.shared_memory.get_search_summaries()
        formatted_chunk_summaries = build_formatted_chunk_summaries(
            chunk_summaries,
            max_content_pieces=self.retrieval_config.final_passages,
            fields=["description"],
        )

        outline_prompt_str = outline_prompt(
            topic, top_chunk_text, formatted_chunk_summaries
        )

        tom_context = self._get_tom_context(
            action="outline_review",
            has_research=bool(top_chunk),
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
            logger.info(
                f"Using generic fallback outline with {len(outline.headings)} sections"
            )

        # Store outline in memory (whether from LLM or fallback)
        self.initial_outline = outline
        self.shared_memory.state["initial_outline"] = outline
        logger.info(
            f"Created outline with {len(outline.headings)} sections: {outline.headings}"
        )
        return state

    # ======================== REFINE OUTLINE NODE ========================
    def _refine_outline(self, state: MinimalState) -> MinimalState:
        """Refine outline node."""
        logger.info("Refining existing outline.")

        try:
            refine_client = self.get_task_client("create_outline")
            topic = self.shared_memory.get_topic()
            top_chunk = self._get_best_chunk()
            top_chunk_text = top_chunk.content if top_chunk else ""
            chunk_summaries = self.shared_memory.get_search_summaries()
            formatted_chunk_summaries = build_formatted_chunk_summaries(
                chunk_summaries,
                max_content_pieces=self.retrieval_config.final_passages,
                fields=["description"],
            )

            tom_context = self._get_tom_context(
                action="outline_review",
                has_research=bool(top_chunk),
                topic=topic,
            )

            # Prompt uses the existing outline and research context
            if self.initial_outline is not None:
                outline = "\n".join(self.initial_outline.headings)
            else:
                logger.warning(
                    "No initial outline found, using empty headings for refinement prompt."
                )
                outline = ""

            refine_prompt = refine_outline_prompt(
                topic, top_chunk_text, formatted_chunk_summaries, outline
            )
            if self._outline_ctx is not None:
                refine_prompt += (
                    f"\n\nPrevious refinement reasoning context: {self._outline_ctx}"
                )

            final_prompt = enhance_prompt_with_tom(refine_prompt, tom_context)
            outline_model: ArticleOutlineValidationModel = (
                refine_client.call_structured_api(
                    prompt=final_prompt, output_schema=ArticleOutlineValidationModel
                )
            )

            refined_outline = Outline(
                title=self.initial_outline.title if self.initial_outline else "",
                headings=outline_model.headings,
            )
            self.shared_memory.update({"initial_outline": refined_outline})
            self.initial_outline = refined_outline

        except Exception as e:
            logger.error(f"Outline refinement failed: {e}", exc_info=True)

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
            has_research=True,
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

        client = self.get_task_client("self_refine_full")
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
        - Collect sections that have PENDING feedback items (current iteration).
        - One LLM call returns WriterValidationBatchModel with per-section updates.
        - Apply status updates + content rewrites.
        """
        iteration = self.shared_memory.state.get("iteration", 0)
        article = self.shared_memory.get_current_draft_as_article()
        if not article:
            raise RuntimeError("Revise node: no current draft in memory.")

        # collect sections -> [pending items]
        pending_by_section = self.shared_memory.get_feedback_items_for_iteration(
            iteration, FeedbackStatus.PENDING
        )
        if not pending_by_section:
            return state  # nothing to do

        mode = getattr(getattr(self, "config", None), "revise_mode", "pending_sections")
        if mode == "single_section":
            self._revise_sections_sequential(article, pending_by_section)
        else:  # default "pending_sections"
            self._revise_sections_batch(article, pending_by_section)

        # persist
        self.shared_memory.update_article_state(article)
        return state

    # endregion Revision Flow Nodes

    # ======================== WRITER V2 HELPERS ========================
    # region WriterV2 Helpers
    def _get_tom_context(
        self, action: str, has_research: bool, topic: str
    ) -> Optional[str]:
        """Generate Theory of Mind context if available."""
        if not (
            hasattr(self.shared_memory, "tom_module")
            and self.shared_memory.tom_module
            and self.shared_memory.tom_module.enabled
        ):
            return None

        try:
            tom_prediction = self.shared_memory.tom_module.predict_agent_response(
                predictor=AgentRole.REVIEWER,
                target=AgentRole.WRITER,
                context={
                    "topic": topic,
                    "has_research": has_research,
                    "action": action,
                },
            )
            return tom_prediction.reasoning
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

            logger.info(
                f"Stored {len(all_results)} final chunks under search_id: {search_id}"
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

    def _get_best_chunk(self) -> Optional[ResearchChunk]:
        """Get the highest-scoring chunk for context."""
        chunks = self.shared_memory.get_stored_chunks()
        if not chunks:
            logger.warning("No chunks available for context")
            return None

        # Sort by relevance score if available, otherwise take first
        sorted_chunks = sorted(
            chunks, key=lambda x: getattr(x, "relevance_score", 0.0), reverse=True
        )

        top_chunk = sorted_chunks[0]
        logger.info(
            f"Selected top chunk: {top_chunk.chunk_id} (score: {getattr(top_chunk, 'relevance_score', 'N/A')})"
        )
        return top_chunk

    def _generate_queries(
        self, topic: str, top_chunk: Optional[ResearchChunk]
    ) -> List[str]:
        """Generate secondary search queries using LLM reasoning."""
        top_chunk_context = top_chunk.content if top_chunk else ""

        prompt = search_query_generation_prompt(
            topic,
            context=top_chunk_context,
            num_queries=self.retrieval_config.num_queries,
        )
        if self._research_ctx is not None:
            prompt += f"\n\nPrevious outline reasoning context: {self._research_ctx}"

        try:
            query_client = self.get_task_client("query_generation")

            # Call the structured API, passing the QueryList model directly
            response_model: QueryListValidationModel = query_client.call_structured_api(
                prompt=prompt, output_schema=QueryListValidationModel
            )
            # Access the validated list directly
            queries = response_model.queries

            logger.info(
                f"Generated {len(queries)} secondary queries via structured output."
            )
            return queries

        except Exception as e:
            logger.error(f"LLM structured query generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate secondary queries: {e}") from e

    def _write_sections(
        self,
        topic: str,
        outline: "Outline",
        chunk_summaries_str: str,
        tom_context: Optional[str],
    ) -> "Article":
        """Sequential section writing (no threads)."""
        sections_out: Dict[str, str] = {}
        for h in outline.headings:
            text = self._write_single_section(
                section=h,
                topic=topic,
                chunk_summaries_str=chunk_summaries_str,
                tom_context=tom_context,
            )
            if text:
                sections_out[h] = text.strip()

        article = Article(
            title=topic,
            content=build_full_article_content(topic, sections_out),
            sections=sections_out,
            metadata={
                "iteration": self.shared_memory.state.get("iteration", 0),
                "sections_count": len(sections_out),
                "writing_mode": "section",
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

        return Article(
            title=topic,
            content=text,
            sections={},  # one-shot; sections can be derived later if desired
            metadata={
                "iteration": self.shared_memory.state.get("iteration", 0),
                "sections_count": len(outline.headings),
                "writing_mode": "full_article",
            },
        )

    def _write_single_section(
        self,
        section: str,
        topic: str,
        chunk_summaries_str: str,
        tom_context: Optional[str] = None,
    ) -> Optional[str]:
        """Single section writer (structured selection + write)."""
        try:
            select_prompt = select_section_chunks_prompt(
                section,
                topic,
                chunk_summaries_str,
                self.retrieval_config.final_passages,
            )
            selection_client = self.get_task_client("section_selection")
            sel = selection_client.call_structured_api(
                prompt=select_prompt, output_schema=ChunkSelectionValidationModel
            )
            chunk_ids = getattr(sel, "chunk_ids", None) or []
            if not chunk_ids:
                return None

            chunk_models = self.shared_memory.get_chunks_by_ids(chunk_ids)
            if not chunk_models:
                return None

            chunks_str = ", ".join(
                f"{cid} {{content: {m.content}, score: {m.metadata.get('relevance_score')}, url: {m.url or 'N/A'}}}"
                for cid, m in chunk_models.items()
            )

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
        # 1) Prompt → batch model
        batch_prompt = build_revision_batch_prompt(article, pending_by_section)
        client = self.get_task_client("revision_batch")
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
        for sec, text in section_patches.items():
            if sec == "_overall":
                # apply to full content only (no full re-write elsewhere)
                article.content = text
            else:
                article.sections[sec] = text

        # rebuild from sections if we have them
        if article.sections:
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

        for section, items in pending_by_section.items():
            # 1) Prompt → single section model
            prompt = build_single_section_revision_prompt(article, section, items)
            vm = client.call_structured_api(
                prompt=prompt, output_schema=WriterValidationModel
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
                # apply to full content only (no full re-write elsewhere)
                article.content = vm.updated_content
            else:
                article.sections = dict(article.sections or {})
                article.sections[section] = vm.updated_content

        # 4) rebuild once at the end → persist once
        if article.sections:
            try:
                article.content = build_full_article_content(
                    article.title, article.sections
                )
            except Exception as e:
                logger.warning(f"[revise/single] Rebuild failed: {e}")

        self.shared_memory.update_article_state(article)


# endregion WriterV2 Helpers
