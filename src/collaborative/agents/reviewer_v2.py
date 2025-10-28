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
        self.strategy_rationale: Optional[str] = None

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
                    item_summary = f"[{section_name}] {item.issue}"

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

            # Persist short summary
            self.verification_summary = verification.summary

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
        Deterministic citation extraction and validation.
        Reads from SharedMemory, stores validation results.
        """
        logger.info("Running fact check (citation validation)...")

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

            # Step 3: Validate citations
            validation_results = validate_citations(
                structured_claims, ref_map, self.shared_memory
            )
            logger.info(
                f"Validation: {validation_results['valid_citations']}/{validation_results['total_citations']} valid, "
                f"{len(validation_results['missing_chunks'])} missing"
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

    # ======================== ANALYZE STRATEGY NODE ========================
    def _analyze_strategy(self, state: MinimalState) -> MinimalState:
        """
        Autonomous planning: determines LLM review strategy.

        Now considers:
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
                # Optionally: model/temperature if your helper supports it
            )

            # Optionally keep these around for later prompt-conditioning
            self.strategy_rationale = decision.rationale

            return decision.strategy

        except Exception as e:
            logger.warning(f"Could not analyze previous feedback: {e}")

        logger.info("Strategy: holistic (fallback)")
        return "holistic"

    def _analyze_previous_feedback(self, items: List[dict]) -> Dict:
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

            # Build strategy-aware prompt
            prompt = build_review_prompt_for_strategy(
                article=article,
                metrics=metrics,
                validation_results=self.validation_results,
                tom_context=self.tom_context,
                strategy=self.review_strategy,
            )
            prompt += """
            IMPORTANT - Quote Field Guidelines:
            - USE quote: When feedback targets specific text (e.g., citation needed for "X showed Y")
            - OMIT quote: When feedback is conceptual or section-level (e.g., "add more depth", "reorganize structure")
            - You can mix: Some items with quotes, some without, based on what makes sense
            """

            review_client = self.get_task_client("review")
            llm_output: ReviewerTaskValidationModel = review_client.call_structured_api(
                prompt=prompt, output_schema=ReviewerTaskValidationModel
            )

            # Store individual feedback items
            items = finalize_feedback_items(llm_output, iteration=self.iteration)
            self.shared_memory.store_feedback_items_for_iteration(self.iteration, items)
            self.reviewer_overall_assessment = llm_output.overall_assessment or ""

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
