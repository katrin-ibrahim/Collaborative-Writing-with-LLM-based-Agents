# src/collaborative/agents/reviewer_v2.py

import logging
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, TypedDict

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    build_review_prompt_for_strategy,
    build_strategy_prompt,
    build_verification_prompt,
)
from src.collaborative.memory.memory import SharedMemory
from src.collaborative.tom.theory_of_mind import AgentRole
from src.collaborative.utils.models import (
    ReviewerTaskValidationModel,
    VerificationValidationModel,
)
from src.collaborative.utils.reviewer_utils import (
    build_ref_map,
    extract_citations,
    finalize_feedback_items,
    get_article_metrics,
    validate_citations,
)
from src.config.config_context import ConfigContext

logger = logging.getLogger(__name__)


StrategyLiteral = Literal[
    "citation-focused", "expansion-focused", "accuracy-focused", "holistic"
]


class StrategyDecision(BaseModel):
    strategy: StrategyLiteral
    rationale: str = Field("", description="Short why-this choice")


# ======================== MINIMAL STATE ========================


class MinimalState(TypedDict):
    """Minimal state for LangGraph compatibility."""


# ======================== REVIEWER AGENT ========================


class ReviewerV2(BaseAgent):
    """
    Reviewer agent methods mapped to LangGraph nodes:
    Graph Flow:
        START → router → [verify_writer_claims (iter 1+)] → fact_check
              → analyze_strategy → generate_feedback → END
    """

    def __init__(self):
        super().__init__()
        self.shared_memory: SharedMemory = ConfigContext.get_memory_instance()
        if not self.shared_memory:
            raise RuntimeError("SharedMemory instance not found in ConfigContext")

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
        self.review_strategy: str = "holistic"
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
        workflow.add_node("verify_writer_claims", self._verify_writer_claims)
        workflow.add_node("fact_check", self._fact_check)
        workflow.add_node("analyze_strategy", self._analyze_strategy)
        workflow.add_node("generate_feedback", self._generate_feedback)

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
        workflow.add_edge("fact_check", "analyze_strategy")
        workflow.add_edge("analyze_strategy", "generate_feedback")
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
                f"Router: VERIFICATION_FLOW (will verify iteration {self.iteration-1} feedback)"
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
                if section_name == "_overall":
                    continue  # Skip overall assessment for verification

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
        Citation extraction, validation, and claim verification.
        Validates both citation existence and claim accuracy against chunk content.
        """
        logger.info("Running fact check (citation validation + claim verification)...")

        try:
            # Get article from SharedMemory
            article = self.shared_memory.get_current_draft_as_article()
            if not article:
                raise RuntimeError("No current draft article found in memory")

            # Step 1: Extract citations
            structured_claims = extract_citations(article)
            logger.info(f"Extracted {len(structured_claims)} citation claims")

            # Step 2: Build reference map
            ref_map = build_ref_map(structured_claims)
            logger.info(f"Built reference map with {len(ref_map)} unique citations")

            # Step 3: Validate citation existence
            validation_results = validate_citations(
                structured_claims, ref_map, self.shared_memory
            )
            logger.info(
                f"Citation existence: {validation_results['valid_citations']}/{validation_results['total_citations']} valid, "
                f"{len(validation_results['missing_chunks'])} missing"
            )

            # Step 4: Verify claims against chunk content (if citations exist)
            if validation_results["valid_citations"] > 0:
                claim_verification = self._verify_claims_against_chunks(
                    structured_claims, ref_map
                )
                validation_results["claim_verification"] = claim_verification
                logger.info(
                    f"Claim verification: checked {claim_verification['total_verified']} claims, "
                    f"found {claim_verification['unsupported_count']} potentially unsupported"
                )

            # Store validation results to SharedMemory
            self.validation_results = validation_results

        except Exception as e:
            logger.error(f"Fact check failed: {e}", exc_info=True)
            self.validation_results = {
                "total_citations": 0,
                "valid_citations": 0,
                "missing_chunks": [],
                "needs_source_count": 0,
            }

        return state

    def _verify_claims_against_chunks(
        self, structured_claims: List, ref_map: Dict
    ) -> Dict:
        """
        Verify claims against cited chunk content using LLM.
        Returns verification results with potentially unsupported claims.
        """
        from src.collaborative.agents.templates import CLAIM_VERIFICATION_PROMPT

        unsupported_claims = []
        total_verified = 0

        # Sample claims to verify (avoid token overflow - max 5 claims)
        sample_size = min(5, len(structured_claims))
        claims_to_verify = (
            structured_claims[:sample_size]
            if len(structured_claims) > sample_size
            else structured_claims
        )

        for claim_obj in claims_to_verify:
            claim_text = claim_obj.get("sentence", "")
            chunk_ids = claim_obj.get("chunks", [])

            if not claim_text or not chunk_ids:
                continue

            total_verified += 1

            # Get chunk content
            chunks = self.shared_memory.get_chunks_by_ids(chunk_ids)
            chunk_contents = []
            for chunk_id, chunk_model in chunks.items():
                chunk_contents.append(
                    f"[Chunk {chunk_id}]: {chunk_model.content[:300]}..."
                )

            if not chunk_contents:
                continue

            # Build verification prompt
            verification_prompt = CLAIM_VERIFICATION_PROMPT.format(
                claim=claim_text, chunks="\n\n".join(chunk_contents)
            )

            try:
                review_client = self.get_task_client("reviewer")
                result = review_client.call_api(prompt=verification_prompt)

                # Check if LLM says claim is unsupported
                if "not supported" in result.lower() or "unsupported" in result.lower():
                    unsupported_claims.append(
                        {
                            "claim": claim_text[:200],
                            "chunk_ids": chunk_ids,
                            "reason": result[:200],
                        }
                    )
                    logger.info(
                        f"Claim verification: UNSUPPORTED claim found - {claim_text[:100]}"
                    )
            except Exception as e:
                logger.warning(f"Claim verification failed for claim: {e}")
                continue

        return {
            "total_verified": total_verified,
            "unsupported_count": len(unsupported_claims),
            "unsupported_claims": unsupported_claims,
        }

    # ======================== ANALYZE STRATEGY NODE ========================
    def _analyze_strategy(self, state: MinimalState) -> MinimalState:
        """
        Autonomous planning: determines LLM review strategy.


        - Article metrics (word count, sections)
        - Citation validation results
        - Previous feedback items and their status (if iteration > 0)
        - Types of unresolved issues
        """
        logger.info("Analyzing article and determining review strategy...")

        try:
            article = self.shared_memory.get_current_draft()

            # Get metrics
            self.metrics = get_article_metrics(article)

            # Determine strategy based on article state AND feedback history
            self.review_strategy = self._determine_review_strategy(
                metrics=self.metrics,
                validation_results=self.validation_results,
                iteration=self.iteration,
            )
            logger.info(f"Selected strategy: {self.review_strategy}")

            # Generate ToM context if available
            self.tom_context = self._get_tom_context(
                self.review_strategy,
                self.metrics,
                self.validation_results,
                self.iteration,
            )
            if self.tom_context:
                logger.info("Generated ToM prediction")

        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}", exc_info=True)
            self.review_strategy = "holistic"

        return state

    def _determine_review_strategy(
        self, metrics: Dict, validation_results: Dict, iteration: int
    ) -> str:
        """
        Determine review approach based on:
        1. Article state (metrics, citations)
        2. Previous feedback history
        3. Unresolved issues from previous iterations
        """
        prev_summary = {
            "has_previous": iteration > 0,
            "pending_counts": {},
            "high_priority_pending": 0,
            "citation_pending": 0,
            "resolution_rate": 0.0,
            "most_common_pending_type": None,
        }
        try:
            if iteration > 0:
                items_by_section = (
                    self.shared_memory.get_feedback_items_for_iteration(
                        iteration=iteration - 1
                    )
                    or {}
                )
                items = []
                for sec, its in items_by_section.items():
                    if sec == "_overall":
                        continue
                    items.extend(its)

                total = len(items) or 1
                pending = [i for i in items if getattr(i, "status", None) == "pending"]
                addressed = [
                    i for i in items if getattr(i, "status", None) == "addressed"
                ]

                # Count by type (robust to missing fields)
                type_counts = {}
                for it in pending:
                    t = getattr(it, "type", None)
                    if t:
                        type_counts[t] = type_counts.get(t, 0) + 1

                prev_summary["pending_counts"] = type_counts
                prev_summary["high_priority_pending"] = sum(
                    1 for it in pending if getattr(it, "priority", None) == "high"
                )
                prev_summary["citation_pending"] = sum(
                    1
                    for it in pending
                    if getattr(it, "type", None)
                    in ("citation_missing", "citation_invalid")
                )
                prev_summary["resolution_rate"] = len(addressed) / total
                prev_summary["most_common_pending_type"] = (
                    max(type_counts, key=lambda k: type_counts[k])
                    if type_counts
                    else None
                )
        except Exception:
            # Non-fatal: keep a minimal summary
            pass

        # Build prompt and call LLM
        try:
            prompt = build_strategy_prompt(
                metrics=metrics,
                validation_results=validation_results,
                iteration=iteration,
                prev_feedback_summary=prev_summary,
            )

            # You already use this pattern elsewhere; reuse the same client style.
            planner_client = self.get_task_client("reviewer")
            decision: StrategyDecision = planner_client.call_structured_api(
                prompt=prompt,
                output_schema=StrategyDecision,
            )

            return decision.strategy

        except Exception as e:
            logger.warning(f"Could not analyze previous feedback: {e}")

        logger.info("Strategy: holistic (fallback)")
        return "holistic"
        """
        Analyze previous feedback items to inform strategy.

        Returns:
            Dict with analysis results:
            - pending_count: Number of pending items
            - pending_high_priority: Number of high-priority pending items
            - pending_citation_items: Citation-related pending items
            - most_common_pending_type: Most frequent type in pending items
            - resolution_rate: Percentage of addressed items
        """
        from collections import Counter

        pending_items = [item for item in items if item.get("status") == "pending"]
        addressed_items = [item for item in items if item.get("status") == "addressed"]

        # Count pending high-priority items
        pending_high_priority = len(
            [item for item in pending_items if item.get("priority") == "high"]
        )

        # Count pending citation items
        pending_citation_items = len(
            [
                item
                for item in pending_items
                if item.get("type") in ["citation_missing", "citation_invalid"]
            ]
        )

        # Find most common type in pending items
        pending_types = [item.get("type") for item in pending_items if item.get("type")]
        most_common_pending_type = None
        if pending_types:
            type_counts = Counter(pending_types)
            most_common_pending_type = type_counts.most_common(1)[0][0]

        # Calculate resolution rate
        total_items = len(items)
        resolution_rate = len(addressed_items) / total_items if total_items > 0 else 0

        analysis = {
            "total_items": total_items,
            "pending_count": len(pending_items),
            "pending_high_priority": pending_high_priority,
            "pending_citation_items": pending_citation_items,
            "most_common_pending_type": most_common_pending_type,
            "resolution_rate": resolution_rate,
        }

        logger.debug(f"Previous feedback analysis: {analysis}")
        return analysis

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
            metrics = get_article_metrics(article.content)

            # Get configuration for reviewer grounding
            from src.config.config_context import ConfigContext

            collab_config = ConfigContext.get_collaboration_config()
            ground_reviewer = getattr(
                collab_config, "ground_reviewer_with_research", True
            )
            max_suggested_queries = getattr(collab_config, "max_suggested_queries", 3)

            # Conditionally include research context
            research_context = None
            if ground_reviewer:
                # Extract CITED chunks only (not all chunks) for fact-checking
                article = self.shared_memory.get_current_draft_as_article()
                if not article:
                    raise RuntimeError("No current draft found")

                structured_claims = extract_citations(article)
                ref_map = build_ref_map(structured_claims)
                cited_chunk_ids = list(ref_map.keys())

                if cited_chunk_ids:
                    # Get only the chunks that were actually cited
                    cited_chunks = self.shared_memory.get_chunks_by_ids(cited_chunk_ids)

                    # Format cited chunks with their content for verification
                    chunk_items = []
                    for chunk_id, chunk_model in cited_chunks.items():
                        chunk_items.append(
                            f"Chunk ID: {chunk_id}\n"
                            f"Content: {chunk_model.content[:500]}...\n"  # Limit to avoid token overflow
                            f"URL: {chunk_model.url or 'N/A'}"
                        )

                    research_context = (
                        "CITED CHUNKS (for fact-checking):\n\n"
                        + "\n\n---\n\n".join(chunk_items)
                    )
                    logger.info(
                        f"Reviewer grounding enabled: providing {len(cited_chunks)} CITED chunks for fact-checking"
                    )
                else:
                    research_context = "No citations found in the article. This is a red flag - the article should cite sources."
                    logger.warning("No citations found - reviewer will flag this")
            else:
                logger.info(
                    "Reviewer grounding disabled: metadata-only review (no research chunks)"
                )

            # Build strategy-aware prompt
            prompt = build_review_prompt_for_strategy(
                article=article,
                metrics=metrics,
                validation_results=self.validation_results,
                tom_context=self.tom_context,
                strategy=self.review_strategy,
                research_context=research_context,
                max_suggested_queries=max_suggested_queries,
            )
            prompt += """
IMPORTANT - Quote Field Guidelines:
- USE quote: When feedback targets specific text (e.g., citation needed for "X showed Y")
- OMIT quote: When feedback is conceptual or section-level (e.g., "add more depth", "reorganize structure")
- You can mix: Some items with quotes, some without, based on what makes sense
"""

            review_client = self.get_task_client("reviewer")
            llm_output: ReviewerTaskValidationModel = review_client.call_structured_api(
                prompt=prompt, output_schema=ReviewerTaskValidationModel
            )

            # Store individual feedback items
            items = finalize_feedback_items(llm_output, iteration=self.iteration)
            self.shared_memory.store_feedback_items_for_iteration(self.iteration, items)
            self.reviewer_overall_assessment = llm_output.overall_assessment or ""

            # Store suggested queries if provided
            if llm_output.suggested_queries:
                # Cap to configured maximum
                suggested_queries = llm_output.suggested_queries[:max_suggested_queries]
                self.shared_memory.store_suggested_queries(
                    self.iteration, suggested_queries
                )
                logger.info(
                    f"Reviewer suggested {len(suggested_queries)} additional queries: {suggested_queries}"
                )
            else:
                logger.info("Reviewer suggested no additional queries")

            logger.info("✓ Stored article-level review data")

        except Exception as e:
            logger.error(f"Feedback generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate review feedback: {e}")

        return state

    # endregion Graph Nodes Definitions

    def _get_tom_context(
        self, strategy: str, metrics: Dict, validation_results: Dict, iteration: int
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
                    "review_strategy": strategy,
                    "article_metrics": metrics,
                    "citation_validation": validation_results,
                    "iteration": iteration,
                    "action": "feedback_response",
                },
            )
            return tom_prediction.reasoning
        except Exception as e:
            logger.warning(f"ToM prediction failed: {e}")
            return None
