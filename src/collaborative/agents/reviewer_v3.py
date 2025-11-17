# src/collaborative/agents/reviewer_v2.py

import logging
from langgraph.graph import END, StateGraph
from typing import Dict, Optional, TypedDict

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.reviewer_templates import (
    build_fact_check_prompt_v2,
    build_review_prompt_v2,
    build_verification_prompt,
)
from src.collaborative.memory.memory import SharedMemory
from src.collaborative.utils.models import (
    FactCheckValidationModel,
    ReviewerTaskValidationModel,
    VerificationValidationModel,
)
from src.collaborative.utils.reviewer_utils import (
    build_reference_map,
    extract_citation_ids,
    finalize_feedback_items,
)
from src.collaborative.utils.writer_utils import build_formatted_chunk_summaries
from src.config.config_context import ConfigContext
from src.utils.category_extractor import CategoryExtractor
from src.utils.infobox_extractor import InfoboxExtractor

logger = logging.getLogger(__name__)


# ======================== MINIMAL STATE ========================


class MinimalState(TypedDict):
    """Minimal state for LangGraph compatibility."""


# ======================== REVIEWER AGENT ========================


class ReviewerV3(BaseAgent):
    """
    Reviewer agent methods mapped to LangGraph nodes:
    Graph Flow:
        START → router → [verify_writer_claims (iter 1+)] → fact_check → generate_feedback → END
    """

    def __init__(self):
        super().__init__()
        self.shared_memory: SharedMemory = ConfigContext.get_memory_instance()
        if not self.shared_memory:
            raise RuntimeError("SharedMemory instance not found in ConfigContext")
        self.retrieval_config = ConfigContext.get_retrieval_config()

        # Initialize instance variables for review cycle state
        self._reset()

        # Build the LangGraph workflow
        self.graph = self._build_graph()
        logger.info("Reviewer agent initialized with LangGraph architecture")

    def _reset(self):
        """Reset instance variables for new review cycle."""
        # Control flow
        self.iteration: int = 0
        self.flow_type: str = ""

        # Intermediate computation results
        self.validation_results: Dict = {}
        self.tom_context: Optional[str] = None
        self.verification_results: Optional[VerificationValidationModel] = None

    def process(self) -> None:
        """Main entry point - initializes state and runs graph."""
        self._reset()
        self.iteration = self.shared_memory.get_iteration()

        logger.info(f"Reviewer starting review for iteration {self.iteration}")

        # Run graph with minimal state (empty dict for API compliance)
        try:
            self.graph.invoke({})
            logger.info(f"Review complete for iteration {self.iteration}")
        except Exception as e:
            logger.error(f"Review workflow failed: {e}", exc_info=True)
            raise

    # ======================== GRAPH CONSTRUCTION ========================
    # region Graph Construction

    def _build_graph(self):
        """Build the LangGraph workflow."""
        workflow = StateGraph(MinimalState)

        # Add nodes
        workflow.add_node("router", self._router_by_iteration)

        # iteration 0 flow (no writer claims exist for us to verify)
        workflow.add_node("fact_check", self._fact_check)
        workflow.add_node("generate_feedback", self._generate_feedback)

        # iteration 1+ flow (verify writer claims from previous iteration, then continue the review
        workflow.add_node("verify_writer_claims", self._verify_writer_claims)

        # Set entry point
        workflow.set_entry_point("router")

        # Conditional edges from router
        workflow.add_conditional_edges(
            "router",
            lambda state: self.flow_type,  # Read from instance variable
            {"INITIAL_FLOW": "fact_check", "VERIFICATION_FLOW": "verify_writer_claims"},
        )

        # Linear flow after routing
        workflow.add_edge("verify_writer_claims", "fact_check")
        workflow.add_edge("fact_check", "generate_feedback")
        workflow.add_edge("generate_feedback", END)

        return workflow.compile()

    # endregion Graph Construction

    # ======================== GRAPH NODES ========================
    # region Graph Nodes Definitions

    # ======================== ROUTER NODE ========================
    def _router_by_iteration(self, state: MinimalState) -> MinimalState:
        """Determines workflow path based on iteration."""
        if self.iteration == 0:
            self.flow_type = "INITIAL_FLOW"
            logger.info("Router: INITIAL_FLOW (no verification needed)")
        else:
            self.flow_type = "VERIFICATION_FLOW"
            logger.info(
                f"Router: VERIFICATION_FLOW (will verify iteration {self.iteration - 1} feedback)"
            )

        return state  # Pass through minimal state

    # ======================== VERIFICATION NODE (Iter 1+ only) ========================
    def _verify_writer_claims(self, state: MinimalState) -> MinimalState:
        """
        LLM-based verification of writer's response to previous feedback.
        Handshake closure - checks if writer addressed feedback.
        """
        previous_iteration = self.iteration - 1
        logger.info(
            f"Verifying writer's response to feedback from iteration {previous_iteration}"
        )

        try:
            # Get current article
            article = self.shared_memory.get_current_draft_as_article()
            if not article:
                raise RuntimeError("No current draft article found in memory")

            # Get previous feedback
            all_items_by_section = self.shared_memory.get_feedback_items_for_iteration(
                iteration=previous_iteration
            )
            if not all_items_by_section:
                logger.warning(f"No feedback found for iteration {previous_iteration}")
                return state

            # Build categorized feedback summary
            addressed_items = []
            pending_items = []
            wont_fix_items = []
            needs_clarification_items = []

            for section_name, items in all_items_by_section.items():
                for item in items:
                    # Include id token so the verifier can reference exact items in structured output
                    item_summary = f"[{section_name}] id={item.id} | {item.issue}"

                    status = item.status
                    if status == "addressed":
                        comment = item.writer_comment
                        addressed_items.append(
                            f"  - {item_summary} (Writer: {comment})"
                        )
                    elif status == "pending":
                        pending_items.append(f"  - {item_summary}")
                    elif status == "wont_fix":
                        reason = item.writer_comment or "No reason given"
                        wont_fix_items.append(f"  - {item_summary} (Reason: {reason})")
                    elif status == "needs_clarification":
                        question = item.writer_comment or ""
                        needs_clarification_items.append(
                            f"  - {item_summary} (Question: {question})"
                        )

            prompt = build_verification_prompt(
                previous_iteration,
                addressed_items=addressed_items,
                pending_items=pending_items,
                wont_fix_items=wont_fix_items,
                needs_clarification_items=needs_clarification_items,
                article=article,
                current_iteration=self.iteration,
            )

            # Call LLM
            review_client = self.get_task_client("reviewer")
            verification: VerificationValidationModel = (
                review_client.call_structured_api(
                    prompt=prompt, output_schema=VerificationValidationModel
                )
            )

            # Apply status flips
            for u in verification.updates:
                self.shared_memory.update_feedback_item_status(u.id, u.status.value)
        except Exception as e:
            logger.error(f"Writer claims verification failed: {e}", exc_info=True)
            # Don't fail the whole workflow - continue to fact checking

        return state

    # ======================== FACT CHECK NODE ========================
    def _fact_check(self, state: MinimalState) -> MinimalState:
        """
        UNIFIED factual verification using BOTH infobox (ground truth) and cited chunks.
        Extracts key claims from article and verifies against both sources.
        """
        logger.info("Running unified factual verification (infobox + chunks)...")

        try:
            # Get article from SharedMemory
            article = self.shared_memory.get_current_draft_as_article()
            if not article:
                raise RuntimeError("No current draft article found in memory")

            # Step 1: Get infobox facts (ground truth)
            topic = self.shared_memory.get_topic()
            infobox_extractor = InfoboxExtractor()
            infobox_data = infobox_extractor.extract_infobox(topic)

            # get cited chunks only via regex from memory
            citation_ids = extract_citation_ids(article.content)
            cited_chunk_contents = self.shared_memory.get_chunks_by_ids(
                citation_ids
            )  # returns Dict[str, ResearchChunk]
            cited_chunks_str = "\n\n".join(
                [chunk.content for chunk in cited_chunk_contents.values()]
            )

            reference_map = build_reference_map(article, citation_ids)

            infobox_str = "\n".join(
                [f"- {key}: {value}" for key, value in infobox_data.items() if value]
            )

            # Step 4: Run unified verification

            prompt = build_fact_check_prompt_v2(
                article=article,
                infobox_data=infobox_str,
                cited_chunks=cited_chunks_str,
                ref_map=reference_map,
            )
            review_client = self.get_task_client("reviewer")
            llm_output: FactCheckValidationModel = review_client.call_structured_api(
                prompt=prompt, output_schema=FactCheckValidationModel
            )
            verification_results = {
                "critical_contradictions": llm_output.critical_contradictions,
                "missing_critical_facts": llm_output.missing_critical_facts,
            }
            if verification_results:
                logger.info(
                    f"Unified factual verification found "
                    f"{len(verification_results['critical_contradictions'])} critical contradictions and "
                    f"{len(verification_results['missing_critical_facts'])} missing critical facts."
                )

                # Log details of critical contradictions
                if verification_results["critical_contradictions"]:
                    logger.debug("Critical contradictions found:")
                    for c in verification_results["critical_contradictions"]:
                        logger.debug(f"  - Section '{c.section}': {c.claim}")
                        logger.debug(f"    Evidence: {c.evidence}")

                # Log details of missing critical facts
                if verification_results["missing_critical_facts"]:
                    logger.debug("Missing critical facts:")
                    for m in verification_results["missing_critical_facts"]:
                        logger.debug(f"  - Section '{m.section}': {m.fact}")
                        logger.debug(f"    Suggested evidence: {m.suggested_evidence}")

            # Log critical findings
            self.validation_results = verification_results

        except Exception as e:
            logger.error(f"Unified factual verification failed: {e}", exc_info=True)
            self.validation_results = {
                "verified_claims": [],
                "critical_contradictions": [],
                "missing_critical_facts": [],
            }

        return state

    # ======================== GENERATE FEEDBACK NODE ========================

    def _generate_feedback(self, state: MinimalState) -> MinimalState:
        """
        LLM semantic review with strategy-aware prompt.
        Uses call_structured_api with Pydantic schema.
        Stores feedback to SharedMemory.
        """
        logger.info("Generating LLM-based review feedback...")

        try:
            article = self.shared_memory.get_current_draft_as_article()
            if not article:
                raise RuntimeError("No current draft article found in memory")

            topic = self.shared_memory.get_topic()

            # chunk_summaries so reviewer can suggest chunks to explore
            chunk_summaries = self.shared_memory.get_search_summaries()
            chunk_summaries_str = build_formatted_chunk_summaries(
                chunk_summaries,
                fields=["description, chunk_id"],
                max_content_pieces=None,
            )

            # fact_check results
            fact_check_results_formatted = ""
            critical_contradictions = self.validation_results.get(
                "critical_contradictions", []
            )
            missing_critical_facts = self.validation_results.get(
                "missing_critical_facts", []
            )

            fact_check_results_formatted += "Critical Contradictions:\n"
            for c in critical_contradictions:
                fact_check_results_formatted += f"- Section: {c.section}\n  Claim: {c.claim}\n  Evidence: {c.evidence}\n"
            fact_check_results_formatted += "\nMissing Critical Facts:\n"
            for m in missing_critical_facts:
                fact_check_results_formatted += f"- Section: {m.section}\n  Fact: {m.fact}\n  Suggested Evidence: {m.suggested_evidence}\n"

            # Potential related articles from categories that the reviewer can suggest to search next
            all_searched_titles = self.shared_memory.get_all_searched_queries()
            category_extractor = CategoryExtractor()

            related_articles = category_extractor.get_related_articles(
                topic, max_categories=15, max_members_per_category=3
            )

            # Just filter the final list against your searched titles
            possible_searches = [
                article
                for article in related_articles
                if article not in all_searched_titles
            ]

            # Get ToM context if enabled
            self.tom_context = self._get_tom_context()

            max_suggested_queries = self.retrieval_config.num_queries
            prompt = build_review_prompt_v2(
                article=article,
                fact_check_results=fact_check_results_formatted,
                chunk_summaries=chunk_summaries_str,
                max_suggested_queries=max_suggested_queries,
                possible_searches=possible_searches,
                tom_context=self.tom_context,
            )

            review_client = self.get_task_client("reviewer")
            llm_output: ReviewerTaskValidationModel = review_client.call_structured_api(
                prompt=prompt, output_schema=ReviewerTaskValidationModel
            )

            if llm_output.suggested_queries:
                # Cap to configured maximum and store structured hints
                suggested_query_hints = llm_output.suggested_queries[
                    : self.retrieval_config.num_queries
                ]
                self.shared_memory["reviewer_suggested_query_hints"] = (
                    suggested_query_hints
                )

                # Extract just the query strings for search execution
                query_strings = [hint.query for hint in suggested_query_hints]
                self.shared_memory["reviewer_suggested_queries"] = query_strings

                logger.info(
                    f"Reviewer suggested {len(suggested_query_hints)} queries with structured hints"
                )
                logger.debug(
                    f"Query hints: {[(h.query, h.intent) for h in suggested_query_hints]}"
                )
            else:
                logger.info("Reviewer suggested no additional queries")

            # Store individual feedback items
            items = finalize_feedback_items(llm_output, iteration=self.iteration)

            logger.info(f"Generated {len(items)} feedback items")

            # Log feedback by type
            if items:
                feedback_by_type = {}
                for item in items:
                    feedback_by_type.setdefault(item.type, []).append(item)

                logger.debug("Feedback breakdown by type:")
                for fb_type, type_items in feedback_by_type.items():
                    logger.debug(f"  {fb_type}: {len(type_items)} items")
                    for item in type_items:
                        logger.debug(f"    - [{item.section}] {item.issue}")

            logger.info("✓ Stored article-level review data")

            # CRITICAL: If LLM generated feedback but ALL items were filtered out due to invalid sections,
            # this is a system failure
            if len(llm_output.items) > 0 and len(items) == 0:
                error_msg = (
                    f"CRITICAL ERROR: Reviewer generated {len(llm_output.items)} feedback items "
                    f"but ALL were filtered out due to invalid section names. "
                    f"This indicates the LLM is not following instructions to use actual section names. "
                    f"The article should NOT be marked as complete."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.shared_memory.store_feedback_items_for_iteration(self.iteration, items)

            logger.info("✓ Stored article-level review data")

        except Exception as e:
            logger.error(f"Feedback generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate review feedback: {e}")

        return state

    # endregion Graph Nodes Definitions

    def _get_tom_context(self) -> Optional[str]:
        """Generate Theory of Mind prediction for writer's likely response to feedback."""
        if not (
            hasattr(self.shared_memory, "tom_module")
            and self.shared_memory.tom_module
            and self.shared_memory.tom_module.enabled
        ):
            return None

        try:
            from src.collaborative.agents.writer_templates import (
                build_reviewer_tom_prediction_prompt,
            )
            from src.collaborative.tom.theory_of_mind import AgentRole

            # Get last observed writer action from memory
            last_writer_action = self.shared_memory.state.get(
                "tom_writer_last_observed_state"
            )

            # Build feedback context (we're about to provide feedback)
            feedback_context = {
                "feedback_count": "pending",  # We don't know yet
                "iteration": self.shared_memory.get_iteration(),
            }

            # Build prediction prompt
            prediction_prompt = build_reviewer_tom_prediction_prompt(
                last_writer_action=last_writer_action,
                feedback_context=feedback_context,
            )

            # Get reviewer's LLM client and make prediction
            client = self.get_task_client("reviewer")
            tom_prediction = self.shared_memory.tom_module.predict_agent_response(
                llm_client=client,
                prompt=prediction_prompt,
            )

            # Store prediction for later evaluation
            self.shared_memory.tom_module.store_prediction(
                predictor_role=AgentRole.REVIEWER,
                target_role=AgentRole.WRITER,
                prediction=tom_prediction,
            )

            # Return reasoning to inject into review prompt
            return tom_prediction.reasoning

        except Exception as e:
            logger.warning(f"ToM prediction failed: {e}")
            return None
