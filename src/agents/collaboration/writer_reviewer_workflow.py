# src/agents/collaboration/writer_reviewer_workflow.py
"""
Advanced writer-reviewer collaboration workflow with iterative improvement.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.agents.reviewer import ReviewerAgent, ReviewFeedback
from src.agents.reviewer.error_handling import validate_reviewer_config
from src.agents.writer.writer_agent import WriterAgent
from src.utils.data import Article

logger = logging.getLogger(__name__)


@dataclass
class CollaborationMetrics:
    """Metrics for tracking collaboration progress."""

    iterations: int
    initial_score: float
    final_score: float
    improvement: float
    convergence_reason: str
    total_time: float
    writer_time: float
    reviewer_time: float
    issues_resolved: int
    recommendations_implemented: int


class WriterReviewerWorkflow:
    """
    Advanced collaboration workflow between WriterAgent and ReviewerAgent.

    Supports iterative improvement, convergence detection, and detailed metrics tracking.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Collaboration parameters
        self.max_iterations = config.get("collaboration.max_iterations", 3)
        self.convergence_threshold = config.get(
            "collaboration.convergence_threshold", 0.05
        )
        self.min_improvement_threshold = config.get(
            "collaboration.min_improvement_threshold", 0.02
        )
        self.target_score = config.get("collaboration.target_score", 0.85)

        # Initialize agents
        writer_config = config.get(
            "writer",
            {
                "writer.max_research_iterations": 2,
                "writer.use_external_knowledge": True,
            },
        )

        reviewer_config = validate_reviewer_config(
            config.get(
                "reviewer",
                {
                    "reviewer.max_claims_per_article": 10,
                    "reviewer.fact_check_timeout": 30,
                    "reviewer.min_claim_confidence": 0.7,
                    "reviewer.enable_structure_analysis": True,
                    "reviewer.feedback_detail_level": "detailed",
                },
            )
        )

        self.writer_agent = WriterAgent(writer_config)
        self.reviewer_agent = ReviewerAgent(reviewer_config)

        # State tracking
        self.collaboration_history = []
        self.metrics = None

    def collaborate(
        self, topic: str
    ) -> Tuple[Article, ReviewFeedback, CollaborationMetrics]:
        """
        Run collaborative writing process with iterative improvement.

        Args:
            topic: Topic to write about

        Returns:
            Tuple of (final_article, final_review, collaboration_metrics)
        """
        import time

        start_time = time.time()
        self.collaboration_history = []

        self.logger.info(f"Starting writer-reviewer collaboration for: {topic}")

        # Initial writing phase
        writer_start = time.time()
        current_article = self.writer_agent.process(topic)
        writer_time = time.time() - writer_start

        # Initial review
        reviewer_start = time.time()
        current_review = self.reviewer_agent.process(current_article)
        reviewer_time = time.time() - reviewer_start

        initial_score = current_review.overall_score
        self.logger.info(f"Initial article score: {initial_score:.3f}")

        # Track collaboration state
        self.collaboration_history.append(
            {
                "iteration": 0,
                "article": current_article,
                "review": current_review,
                "score": initial_score,
                "improvement": 0.0,
            }
        )

        # Iterative improvement loop
        iteration = 1
        convergence_reason = "max_iterations_reached"

        while iteration <= self.max_iterations:
            self.logger.info(f"Starting collaboration iteration {iteration}")

            # Check if target score reached
            if current_review.overall_score >= self.target_score:
                convergence_reason = "target_score_reached"
                self.logger.info(
                    f"Target score {self.target_score} reached: {current_review.overall_score:.3f}"
                )
                break

            # Attempt to improve article based on review feedback
            try:
                improved_article = self._improve_article_with_feedback(
                    current_article, current_review, topic
                )

                if improved_article is None:
                    convergence_reason = "improvement_failed"
                    self.logger.warning(
                        "Failed to improve article, stopping collaboration"
                    )
                    break

                # Review improved article
                reviewer_iter_start = time.time()
                improved_review = self.reviewer_agent.process(improved_article)
                reviewer_time += time.time() - reviewer_iter_start

                # Calculate improvement
                score_improvement = (
                    improved_review.overall_score - current_review.overall_score
                )

                self.logger.info(
                    f"Iteration {iteration}: score {improved_review.overall_score:.3f} (improvement: {score_improvement:+.3f})"
                )

                # Check for convergence
                if abs(score_improvement) < self.convergence_threshold:
                    convergence_reason = "score_converged"
                    self.logger.info(
                        f"Score converged (improvement < {self.convergence_threshold})"
                    )
                    break

                # Check for minimal improvement
                if score_improvement < self.min_improvement_threshold:
                    convergence_reason = "minimal_improvement"
                    self.logger.info(
                        f"Minimal improvement detected (< {self.min_improvement_threshold})"
                    )
                    break

                # Update current state
                current_article = improved_article
                current_review = improved_review

                # Track iteration
                self.collaboration_history.append(
                    {
                        "iteration": iteration,
                        "article": current_article,
                        "review": current_review,
                        "score": current_review.overall_score,
                        "improvement": score_improvement,
                    }
                )

                iteration += 1

            except Exception as e:
                self.logger.error(f"Iteration {iteration} failed: {e}")
                convergence_reason = "iteration_error"
                break

        # Calculate final metrics
        total_time = time.time() - start_time
        final_score = current_review.overall_score
        total_improvement = final_score - initial_score

        # Count resolved issues and implemented recommendations
        issues_resolved = self._count_resolved_issues()
        recommendations_implemented = self._count_implemented_recommendations()

        self.metrics = CollaborationMetrics(
            iterations=len(self.collaboration_history) - 1,  # Exclude initial state
            initial_score=initial_score,
            final_score=final_score,
            improvement=total_improvement,
            convergence_reason=convergence_reason,
            total_time=total_time,
            writer_time=writer_time,
            reviewer_time=reviewer_time,
            issues_resolved=issues_resolved,
            recommendations_implemented=recommendations_implemented,
        )

        self.logger.info(
            f"Collaboration completed: {self.metrics.iterations} iterations, "
            f"score improved from {initial_score:.3f} to {final_score:.3f} "
            f"(+{total_improvement:+.3f}) in {total_time:.2f}s"
        )

        return current_article, current_review, self.metrics

    def _improve_article_with_feedback(
        self, article: Article, review: ReviewFeedback, topic: str
    ) -> Optional[Article]:
        """
        Improve article based on reviewer feedback.

        Args:
            article: Current article
            review: Review feedback
            topic: Original topic

        Returns:
            Improved article or None if improvement failed
        """
        try:
            # Create improvement prompt based on feedback
            improvement_prompt = self._create_improvement_prompt(article, review, topic)

            # Use writer agent's API client to generate improvements
            improved_content = self.writer_agent.call_api(improvement_prompt)

            if not improved_content or len(improved_content.strip()) < 100:
                self.logger.warning("Generated improvement is too short or empty")
                return None

            # Create improved article
            improved_article = Article(
                title=article.title,
                content=improved_content,
                sections=article.sections,  # Keep original sections for now
                metadata={
                    **article.metadata,
                    "improved": True,
                    "improvement_iteration": len(self.collaboration_history),
                    "original_score": review.overall_score,
                },
            )

            return improved_article

        except Exception as e:
            self.logger.error(f"Failed to improve article: {e}")
            return None

    def _create_improvement_prompt(
        self, article: Article, review: ReviewFeedback, topic: str
    ) -> str:
        """Create a prompt for improving the article based on review feedback."""

        # Prioritize critical and major issues
        critical_issues = [
            issue for issue in review.issues if issue.severity.value == "critical"
        ]
        major_issues = [
            issue for issue in review.issues if issue.severity.value == "major"
        ]

        priority_issues = critical_issues + major_issues[:3]  # Top 3 major issues

        issues_text = "\n".join(
            [
                f"- {issue.description} (Suggestion: {issue.suggestion})"
                for issue in priority_issues
            ]
        )

        recommendations_text = "\n".join(
            [f"- {rec}" for rec in review.recommendations[:5]]  # Top 5 recommendations
        )

        prompt = f"""
        Improve the following article about "{topic}" based on the review feedback provided.

        CURRENT ARTICLE:
        {article.content}

        REVIEW SCORE: {review.overall_score:.2f}/1.0

        PRIORITY ISSUES TO ADDRESS:
        {issues_text}

        KEY RECOMMENDATIONS:
        {recommendations_text}

        CATEGORY SCORES:
        {chr(10).join([f"- {cat.value.title()}: {score:.2f}" for cat, score in review.category_scores.items()])}

        Instructions:
        1. Address the priority issues listed above
        2. Implement the key recommendations where possible
        3. Maintain the article's core structure and topic focus
        4. Improve factual accuracy, clarity, and completeness
        5. Ensure smooth transitions between sections
        6. Keep the improved article roughly the same length or longer

        Write the improved article:
        """

        return prompt

    def _count_resolved_issues(self) -> int:
        """Count how many issues were resolved during collaboration."""
        if len(self.collaboration_history) < 2:
            return 0

        initial_issues = len(self.collaboration_history[0]["review"].issues)
        final_issues = len(self.collaboration_history[-1]["review"].issues)

        return max(0, initial_issues - final_issues)

    def _count_implemented_recommendations(self) -> int:
        """Count how many recommendations were implemented (heuristic)."""
        if len(self.collaboration_history) < 2:
            return 0

        # Heuristic: assume recommendations were implemented if scores improved
        total_improvement = 0
        for i in range(1, len(self.collaboration_history)):
            improvement = self.collaboration_history[i]["improvement"]
            if improvement > 0:
                total_improvement += improvement

        # Rough estimate: each 0.1 improvement = 1 recommendation implemented
        return int(total_improvement * 10)

    def get_collaboration_summary(self) -> Dict[str, Any]:
        """Get a summary of the collaboration process."""
        if not self.metrics:
            return {"error": "No collaboration completed yet"}

        return {
            "metrics": {
                "iterations": self.metrics.iterations,
                "initial_score": self.metrics.initial_score,
                "final_score": self.metrics.final_score,
                "improvement": self.metrics.improvement,
                "convergence_reason": self.metrics.convergence_reason,
                "total_time": self.metrics.total_time,
                "issues_resolved": self.metrics.issues_resolved,
                "recommendations_implemented": self.metrics.recommendations_implemented,
            },
            "iteration_history": [
                {
                    "iteration": entry["iteration"],
                    "score": entry["score"],
                    "improvement": entry["improvement"],
                }
                for entry in self.collaboration_history
            ],
            "final_review_summary": (
                self.collaboration_history[-1]["review"].summary
                if self.collaboration_history
                else None
            ),
        }
