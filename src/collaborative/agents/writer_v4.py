# src/collaborative/agents/writer_v2.py
import logging
from langgraph.graph import END, StateGraph
from typing import Any, Dict, List, Optional, TypedDict

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.writer_templates import (
    build_outline_prompt,
    build_research_gate_prompt,
    build_revision_batch_prompt_v2,
    build_write_section_prompt,
    build_write_sections_batch_prompt,
    build_writer_tom_prediction_prompt,
)
from src.collaborative.memory.memory import SharedMemory
from src.collaborative.tom.theory_of_mind import AgentRole
from src.collaborative.utils.models import (
    ArticleOutlineValidationModel,
    BatchSectionWritingModel,
    FeedbackStatus,
    ResearchGateDecision,
    SectionContentModel,
    WriterValidationBatchModel,
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


# ======================== MINIMAL STATE ========================


class MinimalState(TypedDict):
    """Minimal state for LangGraph compatibility."""


# ======================== WRITER AGENT ========================
class WriterV4(BaseAgent):
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

        self._research_retries: int = 0
        self._tried_queries: set = set()
        self._pending_gate_queries: Optional[List[str]] = None
        self.chunk_map: Dict[str, List[str]] = {}  # Chunk selections from outline

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

        # 5. Post-Content Edges
        workflow.add_edge("write", END)
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

            logger.info(f"Research gate decision: {decision.action}")
            logger.debug(f"Reasoning: {decision.reasoning}")
            if decision.new_queries:
                logger.debug(f"New queries suggested: {decision.new_queries}")
            if decision.delete_chunk_ids:
                logger.debug(f"Chunks to delete: {decision.delete_chunk_ids}")

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
        """Outline node: create article outline AND chunk selection in one LLM call."""
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
            fields=["description", "chunk_id"],
        )

        outline_prompt = build_outline_prompt(topic, formatted_chunk_summaries)

        try:
            outline_client = self.get_task_client("outline")
            outline_model = outline_client.call_structured_api(
                prompt=outline_prompt, output_schema=ArticleOutlineValidationModel
            )
            outline = Outline(title=topic, headings=outline_model.headings)

            # Extract and store chunk_map for later use
            self.chunk_map = outline_model.chunk_map
            logger.info(
                f"Outline created with {len(outline.headings)} sections and chunk selections for {len(self.chunk_map)} sections"
            )

            # Debug: Log chunk selections for each section
            for section_heading, chunk_ids in self.chunk_map.items():
                logger.debug(
                    f"Section '{section_heading}' assigned chunks: {chunk_ids}"
                )
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
            # Fallback: no chunk map available
            self.chunk_map = {}
            logger.warning("Fallback outline created without chunk map.")

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

        # Get chunk_map from outline (already selected during outline creation)
        chunk_map = getattr(self, "chunk_map", {})
        headings = outline.headings
        sections_out: Dict[str, str] = {}
        section_summaries: Dict[str, str] = {}

        # Calculate dynamic batch size based on number of sections
        # Keep batch size reasonable: min 3, max 5 sections per batch
        # This ensures good context while avoiding too-small batches
        num_sections = len(headings)

        batch_size = max(
            3, min(5, (num_sections + 4) // 5)
        )  # Between 3-5 sections per batch

        num_batches = (num_sections + batch_size - 1) // batch_size
        logger.info(
            f"Writing {num_sections} sections in {num_batches} batches "
            f"(batch_size={batch_size}) using pre-selected chunk map"
        )

        # Process sections in batches
        writer_client = self.get_task_client("writer")
        for batch_start in range(0, len(headings), batch_size):
            batch_end = min(batch_start + batch_size, len(headings))
            batch_headings = headings[batch_start:batch_end]

            logger.info(
                f"Writing batch {batch_start//batch_size + 1}/{num_batches}: "
                f"sections {batch_start+1}-{batch_end} ({', '.join(batch_headings)})"
            )

            # Prepare sections with their selected chunks
            sections_with_chunks = []
            for heading in batch_headings:
                chunk_ids = chunk_map.get(heading, [])
                logger.debug(f"Section '{heading}' -> chunk_ids from map: {chunk_ids}")

                if chunk_ids:
                    chunk_models = self.shared_memory.get_chunks_by_ids(chunk_ids)
                    if chunk_models:
                        chunks_str = ", ".join(
                            f"{cid} {{content: {m.content}}}"
                            for cid, m in chunk_models.items()
                        )
                        logger.debug(
                            f"Section '{heading}' using {len(chunk_models)} chunks: "
                            f"{list(chunk_models.keys())}"
                        )
                        sections_with_chunks.append(
                            {"section_heading": heading, "relevant_info": chunks_str}
                        )
                        continue

                logger.debug(f"Section '{heading}' has no chunks assigned")
                sections_with_chunks.append(
                    {"section_heading": heading, "relevant_info": ""}
                )

            if not sections_with_chunks:
                continue

            # Get previous summaries for coherence (convert dict to list of dicts)
            previous_summaries_list = None
            if section_summaries:
                previous_summaries_list = [
                    {"section_heading": heading, "summary": summary}
                    for heading, summary in section_summaries.items()
                ]

            # Write batch
            write_prompt = build_write_sections_batch_prompt(
                sections_with_chunks, topic, previous_summaries=previous_summaries_list
            )

            try:
                logger.debug(
                    f"Calling LLM to write batch of {len(sections_with_chunks)} sections"
                )
                batch_result = writer_client.call_structured_api(
                    prompt=write_prompt, output_schema=BatchSectionWritingModel
                )

                logger.debug(
                    f"LLM returned {len(batch_result.sections)} sections in batch result"
                )

                for section_model in batch_result.sections:
                    heading = section_model.section_heading
                    content = section_model.content
                    summary = section_model.summary

                    logger.debug(f"Section '{heading}': content={len(content)} chars")
                    logger.debug(f"  Summary: {summary}")

                    if content and content.strip():
                        sections_out[heading] = content.strip()
                        section_summaries[heading] = summary
                        logger.info(
                            f"  ✓ Written: {heading} ({len(content)} chars, summary: {len(summary)} chars)"
                        )

            except Exception as e:
                logger.warning(f"Batch writing failed: {e}, falling back to sequential")
                # Fallback to sequential writing with structured output

                for item in sections_with_chunks:
                    heading = item["section_heading"]
                    relevant_info = item["relevant_info"]

                    write_prompt = build_write_section_prompt(
                        heading,
                        topic,
                        relevant_info,
                        previous_summaries=previous_summaries_list,
                    )
                    try:
                        section_model = writer_client.call_structured_api(
                            prompt=write_prompt, output_schema=SectionContentModel
                        )
                        if section_model.content and section_model.content.strip():
                            sections_out[heading] = section_model.content.strip()
                            section_summaries[heading] = section_model.summary
                    except Exception:
                        continue

        article = Article(
            title=topic,
            content=build_full_article_content(topic, sections_out),
            sections=sections_out,
            metadata={
                "iteration": self.shared_memory.state.get("iteration", 0),
                "sections_count": len(sections_out),
                "writing_mode": "section_batch",
                "section_summaries": section_summaries,
            },
        )

        self.shared_memory.update_article_state(article)
        return state

    # endregion Initial Flow Nodes

    # ======================== REVISION FLOW NODES ========================
    # region Revision Flow Nodes
    def _revise_draft(self, state: MinimalState) -> MinimalState:
        """
        Revision node - batch section revision with comprehensive feedback context.
        Provides: resolved feedback, ALL pending feedback, search summary, ToM, section summaries.
        """
        iteration = self.shared_memory.get_iteration()
        logger.info(f"Starting revision for iteration {iteration}")

        # Execute reviewer-suggested queries if available
        reviewer_queries = self.shared_memory.state.get("reviewer_suggested_queries")
        if reviewer_queries:
            logger.info(f"Executing {len(reviewer_queries)} reviewer-suggested queries")
            logger.debug(f"Reviewer queries: {reviewer_queries}")
            self._execute_search_phase(reviewer_queries)
            self.shared_memory.state["reviewer_suggested_queries"] = None
            if (
                "all_searched_queries" in self.shared_memory.state
                and self.shared_memory.state["all_searched_queries"] is not None
            ):
                self.shared_memory.state["all_searched_queries"].update(
                    reviewer_queries
                )
            else:
                self.shared_memory.state["all_searched_queries"] = set(reviewer_queries)

        article = self.shared_memory.get_current_draft_as_article()
        if not article:
            raise RuntimeError("Revise node: no current draft in memory.")

        all_pending: Dict[str, List] = {}
        for past_iter in range(iteration):
            iter_pending = self.shared_memory.get_feedback_items_for_iteration(
                past_iter, FeedbackStatus.PENDING
            )
            for section, items in iter_pending.items():
                all_pending.setdefault(section, []).extend(items)

        if not all_pending:
            logger.info("No pending feedback - returning early")
            return state

        logger.info(
            f"Found {sum(len(items) for items in all_pending.values())} pending feedback items "
            f"across {iteration} iterations for {len(all_pending)} sections"
        )

        # Collect resolved feedback from previous iteration for context
        resolved_feedback: Dict[str, List] = {}
        if iteration > 0:
            prev_resolved = self.shared_memory.get_feedback_items_for_iteration(
                iteration - 1, FeedbackStatus.VERIFIED_ADDRESSED
            )
            for section, items in prev_resolved.items():
                resolved_feedback[section] = items
            logger.info(
                f"Found {sum(len(items) for items in resolved_feedback.values())} resolved items for context"
            )

        # Get research context with intelligent filtering
        research_ctx = self._build_revision_research_context(all_pending)

        # Get ToM context
        tom_context = self._get_tom_context()

        # Get section summaries from metadata
        section_summaries = article.metadata.get("section_summaries", {})

        # Calculate dynamic batch size
        num_sections = len(all_pending)
        batch_size = max(
            3, min(5, (num_sections + 4) // 5)
        )  # Between 3-5 sections per batch

        num_batches = (num_sections + batch_size - 1) // batch_size
        logger.info(
            f"Revising {num_sections} sections in {num_batches} batches "
            f"(batch_size={batch_size})"
        )

        # Process sections in batches
        revision_client = self.get_task_client("revision")
        section_list = list(all_pending.items())

        for batch_start in range(0, len(section_list), batch_size):
            batch_end = min(batch_start + batch_size, len(section_list))
            batch_sections = dict(section_list[batch_start:batch_end])
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(section_list) + batch_size - 1) // batch_size

            logger.info(
                f"Revising batch {batch_num}/{total_batches}: "
                f"sections {list(batch_sections.keys())}"
            )

            # Prepare batch data
            sections_to_revise = []
            for section_name, pending_items in batch_sections.items():
                # Get section content and summary
                section_content = article.sections.get(section_name, "")
                section_summary = section_summaries.get(section_name, "")

                # Get resolved feedback for this section
                resolved_items = resolved_feedback.get(section_name, [])

                logger.debug(
                    f"Section '{section_name}': {len(pending_items)} pending, "
                    f"{len(resolved_items)} resolved, content={len(section_content)} chars"
                )

                # Log pending feedback details
                if pending_items:
                    logger.debug(f"  Pending feedback for '{section_name}':")
                    for fb_item in pending_items:
                        logger.debug(f"    - [{fb_item.type}] {fb_item.issue}")

                # Log resolved feedback details
                if resolved_items:
                    logger.debug(f"  Resolved feedback for '{section_name}':")
                    for fb_item in resolved_items:
                        logger.debug(
                            f"    - [{fb_item.type}] {fb_item.issue} [status: {fb_item.status}]"
                        )

                sections_to_revise.append(
                    {
                        "section_name": section_name,
                        "section_content": section_content,
                        "section_summary": section_summary,
                        "pending_feedback": pending_items,
                        "resolved_feedback": resolved_items,
                    }
                )

            # Build revision prompt
            revision_prompt = build_revision_batch_prompt_v2(
                article=article,
                sections_to_revise=sections_to_revise,
                research_context=research_ctx,
                tom_context=tom_context,
            )

            try:

                logger.debug(
                    f"Calling LLM to revise batch of {len(sections_to_revise)} sections"
                )
                batch_result = revision_client.call_structured_api(
                    prompt=revision_prompt, output_schema=WriterValidationBatchModel
                )

                logger.debug(
                    f"LLM returned {len(batch_result.items)} revision items in batch result"
                )

                # Apply results
                logger.debug(
                    f"Processing {len(batch_result.items)} revision items from LLM. "
                    f"Expected {len(batch_sections)} sections in batch."
                )

                for idx, item in enumerate(batch_result.items):
                    section_name = item.section_name

                    logger.debug(
                        f"Processing revision item {idx+1}/{len(batch_result.items)} for section '{section_name}': "
                        f"{len(item.updates)} updates, content={len(item.updated_content)} chars"
                    )

                    # Validate that this section was in the batch
                    if section_name not in batch_sections:
                        logger.warning(
                            f"LLM returned section '{section_name}' which was not in the requested batch: {list(batch_sections.keys())}"
                        )
                        continue

                    # Apply feedback status updates
                    self.shared_memory.apply_feedback_updates(item.updates)

                    # Log status changes with details
                    status_counts = {}
                    for upd in item.updates:
                        status_val = (
                            upd.status
                            if isinstance(upd.status, str)
                            else upd.status.value
                        )
                        status_counts[status_val] = status_counts.get(status_val, 0) + 1

                    logger.debug(f"  Status updates for '{section_name}':")
                    for upd in item.updates:
                        status_val = (
                            upd.status
                            if isinstance(upd.status, str)
                            else upd.status.value
                        )
                        logger.debug(f"    - {upd.id}: {status_val}")

                    logger.info(
                        f"  ✓ Revised {section_name}: {status_counts}, "
                        f"{len(item.updated_content)} chars"
                    )

                    # Update section content
                    article.sections[section_name] = item.updated_content

            except Exception as e:
                logger.error(f"Batch revision failed: {e}", exc_info=True)
                # Continue to next batch instead of failing entirely
                continue

        # Rebuild full article content
        try:
            article.content = build_full_article_content(
                article.title, article.sections
            )
        except Exception as e:
            logger.warning(f"Article rebuild failed: {e}")

        self.shared_memory.update_article_state(article)
        return state

    # endregion Revision Flow Nodes

    # ======================== WRITER V2 HELPERS ========================
    # region WriterV2 Helpers

    def _build_revision_research_context(self, all_pending: Dict[str, List]) -> str:
        """
        Build filtered research context for revision using reviewer query hints.
        Returns full chunk content (not just descriptions) filtered by relevance.
        """
        # Get reviewer's structured query hints if available
        query_hints = self.shared_memory.state.get("reviewer_suggested_query_hints", [])

        # Get all chunks and search summaries
        all_chunks = self.shared_memory.state.get("research_chunks", {})
        search_summaries = self.shared_memory.get_search_summaries()

        # Flatten all chunk results from search summaries
        flattened_chunks = []
        for summary in search_summaries.values():
            flattened_chunks.extend(summary.results)

        # Filter chunks based on query hints if available
        if query_hints:
            filtered_chunk_ids = set()

            for hint in query_hints:
                # Match chunks by keywords or expected fields
                keywords = [hint.query.lower()] + [k.lower() for k in hint.keywords]
                expected_terms = [f.lower() for f in hint.expected_fields]
                all_search_terms = keywords + expected_terms

                for chunk_result in flattened_chunks:
                    chunk_id = chunk_result.get("chunk_id")
                    description = chunk_result.get("description", "").lower()

                    # Check if any search term appears in description
                    if any(term in description for term in all_search_terms):
                        filtered_chunk_ids.add(chunk_id)

            logger.info(
                f"Filtered to from {len(flattened_chunks)},{len(filtered_chunk_ids)} relevant chunks using {len(query_hints)} query hints"
            )
        else:
            # No hints available - use top chunks by relevance
            flattened_chunks.sort(
                key=lambda c: (
                    -c.get("relevance_score", 0.0)
                    if c.get("relevance_score") is not None
                    else c.get("relevance_rank", float("inf"))
                )
            )
            filtered_chunk_ids = {
                c["chunk_id"]
                for c in flattened_chunks[: self.retrieval_config.final_passages]
            }
            logger.info(
                f"No query hints - using top {len(filtered_chunk_ids)} chunks by relevance"
            )

        # Build context string with FULL CONTENT (not just descriptions)
        context_parts = []
        for chunk_id in filtered_chunk_ids:
            chunk = all_chunks.get(chunk_id)
            if chunk:
                content = chunk.get("content", "")
                source = chunk.get("source", "unknown")
                context_parts.append(
                    f"[chunk_id: {chunk_id}] [source: {source}]\nContent: {content}"
                )

        return "\n\n".join(context_parts)

    def _get_tom_context(self) -> Optional[str]:
        """Generate Theory of Mind prediction for reviewer's likely feedback focus."""
        if not (
            hasattr(self.shared_memory, "tom_module")
            and self.shared_memory.tom_module
            and self.shared_memory.tom_module.enabled
        ):
            return None

        try:

            # Build current draft info
            article = self.shared_memory.get_current_draft_as_article()
            current_draft_info = {
                "word_count": len(article.content.split()) if article else 0,
                "section_count": (
                    len(article.sections) if article and article.sections else 0
                ),
                "iteration": self.shared_memory.get_iteration(),
            }

            # Build prediction prompt
            prediction_prompt = build_writer_tom_prediction_prompt(
                current_draft_info=current_draft_info,
                interaction_history=self.shared_memory.state.get(
                    "tom_observation_history", []
                ),
            )

            # Get writer's LLM client and make prediction
            client = self.get_task_client("revision")
            tom_prediction = self.shared_memory.tom_module.predict_agent_response(
                llm_client=client,
                prompt=prediction_prompt,
            )

            # Store prediction for later evaluation
            self.shared_memory.tom_module.store_prediction(
                predictor_role=AgentRole.WRITER,
                target_role=AgentRole.REVIEWER,
                prediction=tom_prediction,
                iteration=self.shared_memory.get_iteration(),
            )

            # Return reasoning to inject into revision prompt
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
                logger.info(
                    f"Retrieved {len(all_results)} results from wiki retrieval manager."
                )
                logger.debug("Search results:")
                for chunk in all_results:
                    logger.debug(f"  - {chunk.chunk_id}: {chunk.description}")
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
            if (
                "all_searched_queries" not in self.shared_memory.state
                or not isinstance(self.shared_memory.state["all_searched_queries"], set)
            ):
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

    # endregion WriterV2 Helpers
