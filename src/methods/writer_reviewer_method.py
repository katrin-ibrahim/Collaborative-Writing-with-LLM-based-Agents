# src/methods/writer_reviewer_method.py
"""
Collaborative writer-reviewer method with iterative improvement.
"""

import time

import logging

from src.collaborative.agents.reviewer_agent import ReviewerAgent
from src.collaborative.agents.writer_agent import WriterAgent
from src.collaborative.memory.memory import SharedMemory
from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.utils.data import Article
from src.utils.data.models import CollaborationMetrics

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

            # Set memory instance for tools to access via ConfigContext
            ConfigContext.set_memory_instance(memory)

            writer = WriterAgent()
            reviewer = ReviewerAgent()

            # Check if we have an existing draft, otherwise create initial draft
            current_draft = memory.get_current_draft()
            if not current_draft:
                logger.info(f"Writer creating initial draft for: {topic}")
                writer_start = time.time()

                # Store topic in shared memory for writer to access
                memory.state.article_content = f"# {topic}\n\n"  # Initial structure

                # Writer creates initial draft using shared memory
                writer.process()

                # Get the created article from shared memory
                created_content = memory.state.article_content
                current_article = Article(
                    title=topic,
                    content=created_content,
                    sections=memory.state.article_sections_by_iteration.get(
                        str(memory.get_current_iteration()), {}
                    ),
                    metadata=memory.state.metadata,
                )

                writer_time += time.time() - writer_start
                memory.store_draft(current_article.content)
            else:
                # Resume from existing draft
                current_article = Article(title=topic, content=current_draft)
                logger.info(f"Resuming collaboration from existing draft")

            # Store article in shared memory for reviewer to access
            memory.state.article_content = current_article.content
            memory.store_article_sections(
                memory.get_current_iteration(), current_article.sections
            )

            # Initial review
            reviewer_start = time.time()
            reviewer.process()  # Now takes no arguments, uses shared memory
            reviewer_time += time.time() - reviewer_start

            # Get review results from shared memory
            review_results = memory.state.metadata.get("review_results", [])
            if review_results:
                latest_review = review_results[-1]  # Get most recent review
                initial_score = latest_review.get("overall_score", 0.0)
            else:
                logger.error("No review results found in shared memory")
                initial_score = 0.0

            collaboration_history = [{"iteration": 0, "score": initial_score}]
            logger.info(f"Initial review score: {initial_score:.3f}")

            # Feedback is already stored by ReviewerAgent during its process() call

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
                if initial_score >= self.convergence_threshold:
                    convergence_reason = "score_threshold_reached"
                    logger.info(
                        f"Convergence reached: score {initial_score:.3f} >= {self.convergence_threshold}"
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
                # Store current article in shared memory for writer to access
                memory.state.article_content = current_article.content
                memory.store_article_sections(
                    memory.get_current_iteration(), current_article.sections
                )

                # Writer now uses shared memory for input and output
                writer.process()

                # Get improved article from shared memory
                improved_content = memory.state.article_content
                improved_article = Article(
                    title=topic,
                    content=improved_content,
                    sections=memory.state.article_sections_by_iteration.get(
                        str(memory.get_current_iteration()), {}
                    ),
                    metadata=memory.state.metadata,
                )
                writer_time += time.time() - writer_start

                if improved_article is None:
                    convergence_reason = "improvement_failed"
                    logger.warning(
                        "Writer failed to improve article, stopping collaboration"
                    )
                    break

                # Store improved draft and update shared memory
                memory.store_draft(improved_article.content)
                memory.state.article_content = improved_article.content
                memory.store_article_sections(
                    memory.get_current_iteration(), improved_article.sections
                )

                # Review improved article
                reviewer_start = time.time()
                reviewer.process()  # Now takes no arguments, uses shared memory
                reviewer_time += time.time() - reviewer_start

                # Get review results from shared memory
                review_results = memory.state.metadata.get("review_results", [])
                if review_results:
                    latest_review = review_results[-1]  # Get most recent review
                    new_score = latest_review.get("overall_score", 0.0)
                    feedback_text = latest_review.get("feedback_text", "")
                else:
                    logger.error("No review results found in shared memory")
                    new_score = 0.0
                    feedback_text = ""

                logger.info(f"📊 REVIEWER ITERATION {iteration} OUTPUT VALIDATION:")
                logger.info(f"  New score: {new_score} (from shared memory)")
                logger.info(f"  New feedback length: {len(feedback_text)} chars")
                issues_count = latest_review.get("issues", [])
                recommendations_count = latest_review.get("recommendations", [])
                logger.info(
                    f"  New issues count: {len(issues_count) if isinstance(issues_count, list) else 0}"
                )
                logger.info(
                    f"  New recommendations count: {len(recommendations_count) if isinstance(recommendations_count, list) else 0}"
                )

                # Feedback is already stored by ReviewerAgent, no need to store again

                # Calculate improvement (get previous score from collaboration_history)
                previous_score = (
                    collaboration_history[-1]["score"] if collaboration_history else 0.0
                )
                score_improvement = new_score - previous_score

                logger.info(
                    f"📈 SCORE IMPROVEMENT: {score_improvement:.3f} (from {previous_score:.3f} to {new_score:.3f})"
                )
                logger.info(
                    f"Iteration {iteration}: score {new_score:.3f} "
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
                collaboration_history.append(
                    {
                        "iteration": iteration,
                        "score": new_score,
                        "improvement": score_improvement,
                    }
                )

            # Calculate final metrics
            total_time = time.time() - start_time

            # Get final score from shared memory
            review_results = memory.state.metadata.get("review_results", [])
            if review_results:
                final_score = review_results[-1].get("overall_score", initial_score)
            else:
                final_score = initial_score

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
                    "review_score": final_score,
                    "review_feedback": (
                        review_results[-1].get("feedback_text", "")
                        if review_results
                        else ""
                    ),
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
