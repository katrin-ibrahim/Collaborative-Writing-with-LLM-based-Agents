# src/methods/writer_reviewer_method.py
"""
Collaborative writer-reviewer method with iterative improvement.
"""

import time

import logging

from src.collaborative.agents.reviewer_agent import ReviewerAgent
from src.collaborative.agents.writer_agent import WriterAgent
from src.collaborative.data_models import CollaborationMetrics
from src.collaborative.memory import SharedMemory
from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.utils.data import Article

logger = logging.getLogger(__name__)


class WriterReviewerMethod(BaseMethod):
    """
    Collaborative method with writer and reviewer agents.

    Workflow:
    1. Writer creates initial draft
    2. Reviewer provides feedback
    3. Writer improves based on feedback
    4. Repeat until convergence or max iterations
    """

    def __init__(self):
        super().__init__()
        self.collab_config = ConfigContext.get_collaboration_config()

        # Extract key parameters for easy access
        self.max_iterations = self.collab_config.max_iterations
        self.convergence_threshold = self.collab_config.convergence_threshold
        self.min_improvement_threshold = self.collab_config.min_improvement_threshold

    def run(self, topic: str) -> Article:
        """
        Generate article using collaborative writer-reviewer process.

        Args:
            topic: Topic to write about

        Returns:
            Final article after collaboration with metadata
        """
        logger.info(f"Running writer-reviewer collaboration for: {topic}")

        start_time = time.time()
        writer_time = 0
        reviewer_time = 0

        try:
            # Initialize shared memory and agents
            memory = SharedMemory(
                topic=topic,
                max_iterations=self.max_iterations,
                min_feedback_threshold=1,
            )
            writer = WriterAgent()
            reviewer = ReviewerAgent()

            # Check if we have an existing draft, otherwise create initial draft
            current_draft = memory.get_current_draft()
            if not current_draft:
                logger.info(f"Writer creating initial draft for: {topic}")
                writer_start = time.time()
                current_article = writer.process(topic)
                writer_time += time.time() - writer_start
                memory.store_draft(current_article.content)
            else:
                # Resume from existing draft
                current_article = Article(title=topic, content=current_draft)
                logger.info(f"Resuming collaboration from existing draft")

            # Initial review
            reviewer_start = time.time()
            current_review = reviewer.process(current_article)
            reviewer_time += time.time() - reviewer_start

            initial_score = current_review.overall_score
            collaboration_history = [{"iteration": 0, "score": initial_score}]

            logger.info(f"Initial review score: {initial_score:.3f}")

            # Add initial feedback to memory
            memory.add_feedback([current_review.feedback_text])

            # Collaboration loop
            convergence_reason = "max_iterations_reached"

            while True:
                # Check convergence using memory system
                converged, reason = memory.check_convergence()
                if converged:
                    convergence_reason = reason
                    logger.info(f"Convergence reached: {reason}")
                    break

                # Check score-based convergence
                if current_review.overall_score >= self.convergence_threshold:
                    convergence_reason = "score_threshold_reached"
                    logger.info(
                        f"Convergence reached: score {current_review.overall_score:.3f} >= {self.convergence_threshold}"
                    )
                    break

                # Move to next iteration
                memory.next_iteration()
                iteration = memory.data["iteration"]

                logger.info(
                    f"Collaboration iteration {iteration}/{self.max_iterations}"
                )

                # Writer improves based on feedback
                writer_start = time.time()
                improved_article = writer.process(
                    topic,
                    previous_article=current_article,
                    review_feedback=current_review,
                )
                writer_time += time.time() - writer_start

                if improved_article is None:
                    convergence_reason = "improvement_failed"
                    logger.warning(
                        "Writer failed to improve article, stopping collaboration"
                    )
                    break

                # Store improved draft
                memory.store_draft(improved_article.content)

                # Review improved article
                reviewer_start = time.time()
                improved_review = reviewer.process(improved_article)
                reviewer_time += time.time() - reviewer_start

                # Add new feedback to memory
                memory.add_feedback([improved_review.feedback_text])

                # Calculate improvement
                score_improvement = (
                    improved_review.overall_score - current_review.overall_score
                )
                logger.info(
                    f"Iteration {iteration}: score {improved_review.overall_score:.3f} "
                    f"(improvement: {score_improvement:+.3f})"
                )

                # Check for minimal improvement
                if score_improvement > self.min_improvement_threshold:
                    convergence_reason = "minimal_improvement"
                    logger.info(
                        f"Minimal improvement detected (< {self.min_improvement_threshold})"
                    )
                    break

                # Update current state
                current_article = improved_article
                current_review = improved_review
                collaboration_history.append(
                    {
                        "iteration": iteration,
                        "score": current_review.overall_score,
                        "improvement": score_improvement,
                    }
                )

            # Calculate final metrics
            total_time = time.time() - start_time
            final_score = current_review.overall_score
            total_improvement = final_score - initial_score

            # Get final convergence reason from memory system (takes priority)
            session_summary = memory.get_session_summary()
            final_convergence_reason = (
                session_summary.get("convergence_reason") or convergence_reason
            )

            # Create collaboration metrics
            metrics = CollaborationMetrics(
                iterations=session_summary["iteration"],
                initial_score=initial_score,
                final_score=final_score,
                improvement=total_improvement,
                total_time=total_time,
                convergence_reason=final_convergence_reason,
            )

            # Update article metadata
            current_article.metadata.update(
                {
                    "method": self.get_method_name(),
                    "collaboration_iterations": metrics.iterations,
                    "initial_score": metrics.initial_score,
                    "final_score": metrics.final_score,
                    "score_improvement": metrics.improvement,
                    "convergence_reason": metrics.convergence_reason,
                    "collaboration_time": metrics.total_time,
                    "writer_time": writer_time,
                    "reviewer_time": reviewer_time,
                    "review_score": current_review.overall_score,
                    "review_feedback": current_review.feedback_text,
                    "session_id": session_summary["session_id"],
                    "draft_version": session_summary["draft_version"],
                }
            )

            logger.info(
                f"Collaboration completed for {topic}: {metrics.iterations} iterations, "
                f"score {initial_score:.3f}→{final_score:.3f} (+{total_improvement:+.3f}) "
                f"in {total_time:.2f}s"
            )

            return current_article

        except Exception as e:
            logger.error(f"Writer-reviewer collaboration failed for '{topic}': {e}")
            from src.utils.article import error_article

            return error_article(topic, str(e), self.get_method_name())
