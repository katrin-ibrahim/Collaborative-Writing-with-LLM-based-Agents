# src/evaluation/review_metrics.py
"""
Review-specific evaluation metrics for ReviewerAgent and collaboration workflows.

This module provides metrics for evaluating review quality, fact-checking accuracy,
and collaboration effectiveness.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.agents.collaboration.writer_reviewer_workflow import CollaborationMetrics
from src.agents.reviewer.data_models import FeedbackCategory, ReviewFeedback
from src.utils.data import Article

logger = logging.getLogger(__name__)


@dataclass
class ReviewEvaluationResult:
    """Comprehensive evaluation result for review quality."""

    review_accuracy: float  # How accurate the review was
    feedback_quality: float  # Quality of generated feedback
    fact_check_precision: float  # Precision of fact-checking
    fact_check_recall: float  # Recall of fact-checking
    structure_analysis_accuracy: float  # Accuracy of structure analysis
    issue_identification_score: float  # How well issues were identified
    recommendation_quality: float  # Quality of recommendations
    overall_review_score: float  # Composite review quality score

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "review_accuracy": self.review_accuracy,
            "feedback_quality": self.feedback_quality,
            "fact_check_precision": self.fact_check_precision,
            "fact_check_recall": self.fact_check_recall,
            "structure_analysis_accuracy": self.structure_analysis_accuracy,
            "issue_identification_score": self.issue_identification_score,
            "recommendation_quality": self.recommendation_quality,
            "overall_review_score": self.overall_review_score,
        }


@dataclass
class CollaborationEvaluationResult:
    """Evaluation result for writer-reviewer collaboration."""

    improvement_effectiveness: float  # How much the collaboration improved the article
    iteration_efficiency: float  # How efficiently iterations were used
    convergence_quality: float  # Quality of convergence detection
    time_efficiency: float  # Time efficiency of collaboration
    issue_resolution_rate: float  # Rate of issue resolution
    recommendation_implementation_rate: float  # Rate of recommendation implementation
    overall_collaboration_score: float  # Composite collaboration score

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            "improvement_effectiveness": self.improvement_effectiveness,
            "iteration_efficiency": self.iteration_efficiency,
            "convergence_quality": self.convergence_quality,
            "time_efficiency": self.time_efficiency,
            "issue_resolution_rate": self.issue_resolution_rate,
            "recommendation_implementation_rate": self.recommendation_implementation_rate,
            "overall_collaboration_score": self.overall_collaboration_score,
        }


class ReviewMetricsCalculator:
    """Calculator for review-specific evaluation metrics."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_review_quality(
        self,
        review_feedback: ReviewFeedback,
        article: Article,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> ReviewEvaluationResult:
        """
        Evaluate the quality of a review.

        Args:
            review_feedback: The review feedback to evaluate
            article: The article that was reviewed
            ground_truth: Optional ground truth data for comparison

        Returns:
            ReviewEvaluationResult with quality metrics
        """
        try:
            # Calculate individual metrics
            review_accuracy = self._calculate_review_accuracy(
                review_feedback, ground_truth
            )
            feedback_quality = self._calculate_feedback_quality(review_feedback)
            fact_check_precision, fact_check_recall = (
                self._calculate_fact_check_metrics(review_feedback, ground_truth)
            )
            structure_analysis_accuracy = self._calculate_structure_analysis_accuracy(
                review_feedback, article
            )
            issue_identification_score = self._calculate_issue_identification_score(
                review_feedback
            )
            recommendation_quality = self._calculate_recommendation_quality(
                review_feedback
            )

            # Calculate overall score
            overall_score = self._calculate_overall_review_score(
                review_accuracy,
                feedback_quality,
                fact_check_precision,
                fact_check_recall,
                structure_analysis_accuracy,
                issue_identification_score,
                recommendation_quality,
            )

            return ReviewEvaluationResult(
                review_accuracy=review_accuracy,
                feedback_quality=feedback_quality,
                fact_check_precision=fact_check_precision,
                fact_check_recall=fact_check_recall,
                structure_analysis_accuracy=structure_analysis_accuracy,
                issue_identification_score=issue_identification_score,
                recommendation_quality=recommendation_quality,
                overall_review_score=overall_score,
            )

        except Exception as e:
            self.logger.error(f"Review quality evaluation failed: {e}")
            return self._get_default_review_evaluation()

    def evaluate_collaboration_effectiveness(
        self,
        collaboration_metrics: CollaborationMetrics,
        initial_article: Article,
        final_article: Article,
        initial_review: ReviewFeedback,
        final_review: ReviewFeedback,
    ) -> CollaborationEvaluationResult:
        """
        Evaluate the effectiveness of writer-reviewer collaboration.

        Args:
            collaboration_metrics: Metrics from the collaboration process
            initial_article: Article before collaboration
            final_article: Article after collaboration
            initial_review: Initial review feedback
            final_review: Final review feedback

        Returns:
            CollaborationEvaluationResult with effectiveness metrics
        """
        try:
            # Calculate individual metrics
            improvement_effectiveness = self._calculate_improvement_effectiveness(
                collaboration_metrics
            )
            iteration_efficiency = self._calculate_iteration_efficiency(
                collaboration_metrics
            )
            convergence_quality = self._calculate_convergence_quality(
                collaboration_metrics
            )
            time_efficiency = self._calculate_time_efficiency(collaboration_metrics)
            issue_resolution_rate = self._calculate_issue_resolution_rate(
                initial_review, final_review
            )
            recommendation_implementation_rate = (
                self._calculate_recommendation_implementation_rate(
                    collaboration_metrics, initial_review
                )
            )

            # Calculate overall score
            overall_score = self._calculate_overall_collaboration_score(
                improvement_effectiveness,
                iteration_efficiency,
                convergence_quality,
                time_efficiency,
                issue_resolution_rate,
                recommendation_implementation_rate,
            )

            return CollaborationEvaluationResult(
                improvement_effectiveness=improvement_effectiveness,
                iteration_efficiency=iteration_efficiency,
                convergence_quality=convergence_quality,
                time_efficiency=time_efficiency,
                issue_resolution_rate=issue_resolution_rate,
                recommendation_implementation_rate=recommendation_implementation_rate,
                overall_collaboration_score=overall_score,
            )

        except Exception as e:
            self.logger.error(f"Collaboration effectiveness evaluation failed: {e}")
            return self._get_default_collaboration_evaluation()

    def _calculate_review_accuracy(
        self, review_feedback: ReviewFeedback, ground_truth: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate how accurate the review was compared to ground truth."""
        if not ground_truth:
            # Without ground truth, use internal consistency metrics
            return self._calculate_internal_review_consistency(review_feedback)

        # Compare against ground truth if available
        gt_score = ground_truth.get("expected_score", 0.5)
        score_accuracy = 1.0 - abs(review_feedback.overall_score - gt_score)

        return max(0.0, min(1.0, score_accuracy))

    def _calculate_internal_review_consistency(
        self, review_feedback: ReviewFeedback
    ) -> float:
        """Calculate internal consistency of the review."""
        # Check if overall score aligns with category scores
        category_avg = sum(review_feedback.category_scores.values()) / len(
            review_feedback.category_scores
        )
        score_consistency = 1.0 - abs(review_feedback.overall_score - category_avg)

        # Check if number of issues aligns with scores
        issue_severity_score = self._calculate_issue_severity_score(
            review_feedback.issues
        )
        issue_consistency = 1.0 - abs(
            review_feedback.overall_score - (1.0 - issue_severity_score)
        )

        return (score_consistency + issue_consistency) / 2

    def _calculate_issue_severity_score(self, issues: List) -> float:
        """Calculate severity score based on issues found."""
        if not issues:
            return 0.0

        severity_weights = {
            "critical": 1.0,
            "major": 0.7,
            "minor": 0.3,
            "suggestion": 0.1,
        }

        total_severity = sum(
            severity_weights.get(issue.severity.value, 0.5) for issue in issues
        )

        # Normalize by number of issues
        return min(1.0, total_severity / len(issues))

    def _calculate_feedback_quality(self, review_feedback: ReviewFeedback) -> float:
        """Calculate quality of feedback provided."""
        quality_score = 0.0

        # Check if feedback has actionable suggestions
        actionable_issues = sum(
            1 for issue in review_feedback.issues if issue.suggestion
        )
        if review_feedback.issues:
            quality_score += 0.3 * (actionable_issues / len(review_feedback.issues))
        else:
            quality_score += 0.3  # No issues is good

        # Check if recommendations are provided
        if review_feedback.recommendations:
            quality_score += 0.3

        # Check if summary is informative
        if review_feedback.summary and len(review_feedback.summary) > 20:
            quality_score += 0.2

        # Check category coverage
        expected_categories = {
            FeedbackCategory.FACTUAL,
            FeedbackCategory.STRUCTURAL,
            FeedbackCategory.CLARITY,
            FeedbackCategory.COMPLETENESS,
        }
        covered_categories = set(review_feedback.category_scores.keys())
        category_coverage = len(
            covered_categories.intersection(expected_categories)
        ) / len(expected_categories)
        quality_score += 0.2 * category_coverage

        return min(1.0, quality_score)

    def _calculate_fact_check_metrics(
        self, review_feedback: ReviewFeedback, ground_truth: Optional[Dict[str, Any]]
    ) -> Tuple[float, float]:
        """Calculate fact-checking precision and recall."""
        if not ground_truth or "fact_check_results" not in review_feedback.metadata:
            # Without ground truth, use heuristic based on verification status
            fact_check_results = review_feedback.metadata.get("fact_check_results", [])
            if not fact_check_results:
                return 0.5, 0.5  # Default values

            verified_count = sum(
                1
                for r in fact_check_results
                if r.get("verification_status") == "verified"
            )
            total_count = len(fact_check_results)

            precision = verified_count / total_count if total_count > 0 else 0.0
            recall = 0.7 if verified_count > 0 else 0.3  # Heuristic

            return precision, recall

        # Calculate against ground truth
        gt_facts = ground_truth.get("known_facts", [])
        identified_facts = review_feedback.metadata.get("claims_analyzed", 0)

        precision = min(1.0, identified_facts / max(1, len(gt_facts)))
        recall = min(1.0, len(gt_facts) / max(1, identified_facts))

        return precision, recall

    def _calculate_structure_analysis_accuracy(
        self, review_feedback: ReviewFeedback, article: Article
    ) -> float:
        """Calculate accuracy of structure analysis."""
        # Heuristic based on article characteristics
        content_length = len(article.content)
        sections_count = len(article.sections)

        # Expected structure score based on article characteristics
        expected_structure_score = min(
            1.0, (content_length / 1000) * 0.5 + (sections_count / 5) * 0.5
        )

        # Get actual structure score from review
        actual_structure_score = review_feedback.category_scores.get(
            FeedbackCategory.STRUCTURAL, 0.5
        )

        # Calculate accuracy as inverse of difference
        accuracy = 1.0 - abs(expected_structure_score - actual_structure_score)

        return max(0.0, min(1.0, accuracy))

    def _calculate_issue_identification_score(
        self, review_feedback: ReviewFeedback
    ) -> float:
        """Calculate how well issues were identified."""
        if not review_feedback.issues:
            return 0.8  # No issues found might be correct

        # Score based on issue diversity and specificity
        categories_covered = set(issue.category for issue in review_feedback.issues)
        category_diversity = len(categories_covered) / len(FeedbackCategory)

        # Score based on specificity of descriptions
        specific_issues = sum(
            1 for issue in review_feedback.issues if len(issue.description) > 20
        )
        specificity_score = specific_issues / len(review_feedback.issues)

        return (category_diversity + specificity_score) / 2

    def _calculate_recommendation_quality(
        self, review_feedback: ReviewFeedback
    ) -> float:
        """Calculate quality of recommendations."""
        if not review_feedback.recommendations:
            return 0.3  # No recommendations is poor

        # Score based on number and length of recommendations
        quality_score = min(
            1.0, len(review_feedback.recommendations) / 5
        )  # Up to 5 recommendations

        # Bonus for detailed recommendations
        detailed_recs = sum(
            1 for rec in review_feedback.recommendations if len(rec) > 30
        )
        if review_feedback.recommendations:
            quality_score += 0.3 * (
                detailed_recs / len(review_feedback.recommendations)
            )

        return min(1.0, quality_score)

    def _calculate_overall_review_score(self, *scores) -> float:
        """Calculate overall review quality score."""
        weights = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]  # Weights for each metric
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        return min(1.0, max(0.0, weighted_score))

    def _calculate_improvement_effectiveness(
        self, metrics: CollaborationMetrics
    ) -> float:
        """Calculate how effectively the collaboration improved the article."""
        if metrics.improvement <= 0:
            return 0.0

        # Score based on improvement magnitude and efficiency
        improvement_score = min(
            1.0, metrics.improvement * 2
        )  # 0.5 improvement = perfect score

        # Bonus for reaching high final scores
        if metrics.final_score >= 0.8:
            improvement_score += 0.2

        return min(1.0, improvement_score)

    def _calculate_iteration_efficiency(self, metrics: CollaborationMetrics) -> float:
        """Calculate efficiency of iteration usage."""
        if metrics.iterations == 0:
            return 1.0  # Perfect if no iterations needed

        # Score based on improvement per iteration
        improvement_per_iteration = metrics.improvement / metrics.iterations
        efficiency_score = min(
            1.0, improvement_per_iteration * 10
        )  # 0.1 per iteration = perfect

        return efficiency_score

    def _calculate_convergence_quality(self, metrics: CollaborationMetrics) -> float:
        """Calculate quality of convergence detection."""
        convergence_scores = {
            "target_score_reached": 1.0,
            "score_converged": 0.8,
            "minimal_improvement": 0.6,
            "max_iterations_reached": 0.4,
            "improvement_failed": 0.2,
            "iteration_error": 0.1,
        }

        return convergence_scores.get(metrics.convergence_reason, 0.5)

    def _calculate_time_efficiency(self, metrics: CollaborationMetrics) -> float:
        """Calculate time efficiency of collaboration."""
        if metrics.total_time <= 0:
            return 0.0

        # Score based on improvement per unit time
        improvement_per_second = metrics.improvement / metrics.total_time
        efficiency_score = min(
            1.0, improvement_per_second * 100
        )  # 0.01 per second = perfect

        return efficiency_score

    def _calculate_issue_resolution_rate(
        self, initial_review: ReviewFeedback, final_review: ReviewFeedback
    ) -> float:
        """Calculate rate of issue resolution."""
        initial_issues = len(initial_review.issues)
        final_issues = len(final_review.issues)

        if initial_issues == 0:
            return 1.0  # No issues to resolve

        resolved_issues = max(0, initial_issues - final_issues)
        resolution_rate = resolved_issues / initial_issues

        return resolution_rate

    def _calculate_recommendation_implementation_rate(
        self, metrics: CollaborationMetrics, initial_review: ReviewFeedback
    ) -> float:
        """Calculate rate of recommendation implementation."""
        initial_recommendations = len(initial_review.recommendations)

        if initial_recommendations == 0:
            return 1.0  # No recommendations to implement

        # Use heuristic based on improvement
        implementation_rate = min(
            1.0, metrics.improvement * 2
        )  # 0.5 improvement = all implemented

        return implementation_rate

    def _calculate_overall_collaboration_score(self, *scores) -> float:
        """Calculate overall collaboration effectiveness score."""
        weights = [0.25, 0.15, 0.15, 0.1, 0.2, 0.15]  # Weights for each metric
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        return min(1.0, max(0.0, weighted_score))

    def _get_default_review_evaluation(self) -> ReviewEvaluationResult:
        """Get default review evaluation for error cases."""
        return ReviewEvaluationResult(
            review_accuracy=0.5,
            feedback_quality=0.5,
            fact_check_precision=0.5,
            fact_check_recall=0.5,
            structure_analysis_accuracy=0.5,
            issue_identification_score=0.5,
            recommendation_quality=0.5,
            overall_review_score=0.5,
        )

    def _get_default_collaboration_evaluation(self) -> CollaborationEvaluationResult:
        """Get default collaboration evaluation for error cases."""
        return CollaborationEvaluationResult(
            improvement_effectiveness=0.5,
            iteration_efficiency=0.5,
            convergence_quality=0.5,
            time_efficiency=0.5,
            issue_resolution_rate=0.5,
            recommendation_implementation_rate=0.5,
            overall_collaboration_score=0.5,
        )


# Convenience functions for easy integration
def evaluate_review_quality(
    review_feedback: ReviewFeedback,
    article: Article,
    ground_truth: Optional[Dict[str, Any]] = None,
) -> ReviewEvaluationResult:
    """Convenience function for evaluating review quality."""
    calculator = ReviewMetricsCalculator()
    return calculator.evaluate_review_quality(review_feedback, article, ground_truth)


def evaluate_collaboration_effectiveness(
    collaboration_metrics: CollaborationMetrics,
    initial_article: Article,
    final_article: Article,
    initial_review: ReviewFeedback,
    final_review: ReviewFeedback,
) -> CollaborationEvaluationResult:
    """Convenience function for evaluating collaboration effectiveness."""
    calculator = ReviewMetricsCalculator()
    return calculator.evaluate_collaboration_effectiveness(
        collaboration_metrics,
        initial_article,
        final_article,
        initial_review,
        final_review,
    )
