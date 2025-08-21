# src/methods/writer_reviewer_method.py
"""
Collaborative writer-reviewer method with iterative improvement.
"""

import time

import logging

from src.collaborative.agents.reviewer_agent import ReviewerAgent
from src.collaborative.agents.writer_agent import WriterAgent
from src.collaborative.data_models import CollaborationMetrics
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

    def __init__(self, client, retrieval_config, collaboration_config):
        super().__init__(client, retrieval_config, collaboration_config)

        # Collaboration parameters
        self.max_iterations = collaboration_config.get("max_iterations", 3)
        self.convergence_threshold = collaboration_config.get(
            "convergence_threshold", 0.85
        )
        self.min_improvement_threshold = collaboration_config.get(
            "min_improvement_threshold", 0.02
        )

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
            # Initialize agents
            writer = WriterAgent(
                self.retrieval_config, self.collaboration_config, self.model_config
            )
            reviewer = ReviewerAgent(
                self.retrieval_config, self.collaboration_config, self.model_config
            )

            # Initial draft
            logger.info(f"Writer creating initial draft for: {topic}")
            writer_start = time.time()
            current_article = writer.process(topic)
            writer_time += time.time() - writer_start

            # Initial review
            reviewer_start = time.time()
            current_review = reviewer.process(current_article)
            reviewer_time += time.time() - reviewer_start

            initial_score = current_review.overall_score
            collaboration_history = [{"iteration": 0, "score": initial_score}]

            logger.info(f"Initial review score: {initial_score:.3f}")

            # Collaboration loop
            convergence_reason = "max_iterations_reached"

            for iteration in range(1, self.max_iterations + 1):
                logger.info(
                    f"Collaboration iteration {iteration}/{self.max_iterations}"
                )

                # Check convergence
                if current_review.overall_score >= self.convergence_threshold:
                    convergence_reason = "score_threshold_reached"
                    logger.info(
                        f"Convergence reached: score {current_review.overall_score:.3f} >= {self.convergence_threshold}"
                    )
                    break

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

                # Review improved article
                reviewer_start = time.time()
                improved_review = reviewer.process(improved_article)
                reviewer_time += time.time() - reviewer_start

                # Calculate improvement
                score_improvement = (
                    improved_review.overall_score - current_review.overall_score
                )
                logger.info(
                    f"Iteration {iteration}: score {improved_review.overall_score:.3f} "
                    f"(improvement: {score_improvement:+.3f})"
                )

                # Check for minimal improvement
                if score_improvement < self.min_improvement_threshold:
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

            # Create collaboration metrics
            metrics = CollaborationMetrics(
                iterations=len(collaboration_history) - 1,
                initial_score=initial_score,
                final_score=final_score,
                improvement=total_improvement,
                total_time=total_time,
                convergence_reason=convergence_reason,
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
