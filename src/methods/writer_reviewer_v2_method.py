# src/methods/writer_reviewer_v2_method.py

import time

import logging

from src.collaborative.agents.reviewer_v2 import ReviewerV2
from src.collaborative.agents.writer_v2 import WriterV2
from src.collaborative.memory.memory import SharedMemory
from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.utils.data import Article

logger = logging.getLogger(__name__)


class WriterReviewerV2Method(BaseMethod):
    """
    Simplified collaborative method using WriterV2 and ReviewerV2 agents.

    Args:
        tom_enabled: Whether to enable Theory of Mind for inter-agent reasoning

    Workflow:
    1. WriterV2 creates initial draft with research
    2. ReviewerV2 provides fact-checked feedback
    3. WriterV2 revises based on feedback
    4. Repeat until convergence or max iterations
    """

    def __init__(self, tom_enabled: bool = False):
        super().__init__()
        self.collab_config = ConfigContext.get_collaboration_config()
        self.tom_enabled = tom_enabled

        # Extract key parameters with defaults if config is None
        if self.collab_config:
            self.max_iterations = self.collab_config.max_iterations
            self.convergence_threshold = self.collab_config.convergence_threshold
            self.min_improvement_threshold = (
                self.collab_config.min_improvement_threshold
            )
        else:
            # Default collaboration parameters
            self.max_iterations = 3
            self.convergence_threshold = 0.9
            self.min_improvement_threshold = 0.1
            logger.warning("No collaboration config found, using default parameters")

    def run(self, topic: str) -> Article:
        """
        Generate article using WriterV2-ReviewerV2 collaboration.

        Args:
            topic: Topic to write about

        Returns:
            Final article after collaboration with comprehensive metadata
        """
        logger.info(f"Running WriterV2-ReviewerV2 collaboration for: {topic}")
        logger.info(f"Theory of Mind enabled: {self.tom_enabled}")

        start_time = time.time()
        writer_time = 0
        reviewer_time = 0

        try:
            # Initialize shared memory with Theory of Mind support
            memory = SharedMemory(
                topic=topic,
                max_iterations=self.max_iterations,
                min_feedback_threshold=1,
                tom_enabled=self.tom_enabled,
            )

            # Set memory instance for tools and agents to access
            ConfigContext.set_memory_instance(memory)

            # Initialize V2 agents
            writer = WriterV2()
            reviewer = ReviewerV2()

            # Check if we have an existing draft, otherwise create initial draft
            current_draft = memory.get_current_draft()
            if not current_draft:
                logger.info(f"WriterV2 creating initial draft for: {topic}")
                writer_start = time.time()

                # Initialize article state in memory
                seed_article = Article(
                    title=topic,
                    content=f"# {topic}\n\n",
                    sections={},
                    metadata=memory.state.get("metadata", {}).copy(),
                )
                memory.update_article_state(seed_article)

            # Collaborative improvement loop - let convergence checker handle all termination logic
            while True:
                # WriterV2 processes (initial draft or revision based on iteration)
                try:
                    current_iteration = memory.get_current_iteration()
                    if current_iteration == 0:
                        logger.info("WriterV2 creating initial draft")
                    else:
                        logger.info("WriterV2 addressing feedback and revising")

                    writer.process()
                    writer_time += time.time() - writer_start
                    current_draft = memory.get_current_draft()

                    if not current_draft:
                        raise RuntimeError("WriterV2 failed to create draft")

                    if current_iteration == 0:
                        logger.info(
                            f"WriterV2 completed initial draft: {len(current_draft)} characters"
                        )
                    else:
                        logger.info(
                            f"WriterV2 completed revision: {len(current_draft)} characters"
                        )

                    # ReviewerV2 analyzes and provides feedback
                    logger.info("ReviewerV2 analyzing article and generating feedback")
                    reviewer_start = time.time()

                    reviewer.process()
                    reviewer_time += time.time() - reviewer_start
                    logger.info("ReviewerV2 completed analysis and feedback generation")
                except Exception as e:
                    current_iteration = memory.get_current_iteration()
                    logger.error(f"V2 failed in iteration {current_iteration}: {e}")
                    raise RuntimeError(f"Workflow failed: {e}") from e

                # Check for convergence BEFORE incrementing iteration
                logger.info("Checking convergence criteria")
                converged, convergence_reason = memory.check_convergence()
                if converged:
                    logger.info(f"Convergence achieved: {convergence_reason}")
                    break

                # Move to next iteration for writer revision
                memory.next_iteration()
                writer_start = time.time()

            # Prepare final article with comprehensive metadata
            final_article = memory.get_current_article()
            if not final_article:
                raise RuntimeError("No final article available")

            # Enhanced metadata for WriterV2-ReviewerV2 method
            total_time = time.time() - start_time

            # Get collaboration metrics
            collab_metrics = self._calculate_collaboration_metrics(
                memory, writer_time, reviewer_time, total_time
            )

            # Enhanced metadata
            final_iterations = memory.get_current_iteration()
            final_article.metadata.update(
                {
                    # Standard metadata fields expected by output_manager
                    "generation_time": total_time,
                    "model": getattr(writer.api_client, "model_path", "unknown"),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    # Method-specific metadata
                    "method": "writer_reviewer_v2",
                    "tom_enabled": self.tom_enabled,
                    "total_iterations": final_iterations,
                    "converged": memory.state.get("converged", False),
                    "convergence_reason": memory.state.get("convergence_reason"),
                    "execution_time": {
                        "total_time": total_time,
                        "writer_time": writer_time,
                        "reviewer_time": reviewer_time,
                        "overhead_time": total_time - writer_time - reviewer_time,
                    },
                    "collaboration_metrics": collab_metrics,
                    "theory_of_mind": (
                        self._get_tom_metrics(memory) if self.tom_enabled else None
                    ),
                    "workflow_type": "hybrid_deterministic",
                    "agent_versions": {"writer": "v2", "reviewer": "v2"},
                }
            )

            logger.info(f"WriterV2-ReviewerV2 collaboration completed successfully")
            logger.info(
                f"Final article: {len(final_article.content)} characters, {len(final_article.sections)} sections"
            )
            logger.info(
                f"Total time: {total_time:.2f}s, Iterations: {final_iterations}, Converged: {final_article.metadata['converged']}"
            )

            return final_article

        except Exception as e:
            logger.error(
                f"WriterV2-ReviewerV2 collaboration failed: {e}", exc_info=True
            )
            raise RuntimeError(f"WriterReviewerV2Method failed: {e}") from e

    def _calculate_collaboration_metrics(
        self,
        memory: SharedMemory,
        writer_time: float,
        reviewer_time: float,
        total_time: float,
    ) -> dict:
        """Calculate comprehensive collaboration metrics."""

        # Get feedback statistics
        all_feedback = memory.get_review_feedback()
        feedback_by_iteration = {}

        for feedback in all_feedback:
            iteration = feedback.get("iteration", 0)
            if iteration not in feedback_by_iteration:
                feedback_by_iteration[iteration] = []
            feedback_by_iteration[iteration].append(feedback)

        # Calculate feedback metrics
        total_feedback = len(all_feedback)
        addressed_feedback = len(
            [f for f in all_feedback if f.get("status") == "addressed"]
        )
        pending_feedback = len(
            [f for f in all_feedback if f.get("status") == "pending"]
        )

        # Calculate convergence metrics using existing checker
        convergence_metrics = {}
        if hasattr(memory, "convergence_checker"):
            try:
                convergence_data = memory.convergence_checker.check_convergence(
                    memory.state
                )
                convergence_metrics = {
                    "feedback_addressed_percentage": convergence_data.get(
                        "feedback_addressed_percentage", 0
                    ),
                    "convergence_score": convergence_data.get("convergence_score", 0),
                    "convergence_criteria_met": convergence_data.get(
                        "converged", False
                    ),
                }
            except Exception as e:
                logger.warning(f"Could not calculate convergence metrics: {e}")

        return {
            "total_feedback_items": total_feedback,
            "addressed_feedback": addressed_feedback,
            "pending_feedback": pending_feedback,
            "feedback_resolution_rate": (
                addressed_feedback / total_feedback if total_feedback > 0 else 0
            ),
            "feedback_by_iteration": {
                str(k): len(v) for k, v in feedback_by_iteration.items()
            },
            "average_feedback_per_iteration": total_feedback
            / max(1, len(feedback_by_iteration)),
            "time_efficiency": {
                "writer_time_percentage": (
                    writer_time / total_time if total_time > 0 else 0
                ),
                "reviewer_time_percentage": (
                    reviewer_time / total_time if total_time > 0 else 0
                ),
                "seconds_per_iteration": total_time
                / max(1, memory.get_current_iteration()),
            },
            "convergence": convergence_metrics,
        }

    def _get_tom_metrics(self, memory: SharedMemory) -> dict:
        """Get Theory of Mind metrics if available."""
        if not memory.tom_module or not memory.tom_module.enabled:
            return None

        try:
            tom_interactions = memory.tom_module.interactions
            if not tom_interactions:
                return {"interactions": 0, "predictions": 0, "accuracy": 0}

            total_predictions = 0
            accurate_predictions = 0

            for interaction in tom_interactions:
                # Count writer predictions
                for pred in interaction.writer_predictions:
                    total_predictions += 1
                    if pred.accuracy.value == "correct":
                        accurate_predictions += 1

                # Count reviewer predictions
                for pred in interaction.reviewer_predictions:
                    total_predictions += 1
                    if pred.accuracy.value == "correct":
                        accurate_predictions += 1

            return {
                "interactions": len(tom_interactions),
                "total_predictions": total_predictions,
                "accurate_predictions": accurate_predictions,
                "accuracy_rate": (
                    accurate_predictions / total_predictions
                    if total_predictions > 0
                    else 0
                ),
                "learning_occurred": sum(
                    1 for i in tom_interactions if i.learning_occurred
                ),
            }

        except Exception as e:
            logger.warning(f"Could not calculate ToM metrics: {e}")
            return {"error": str(e)}
