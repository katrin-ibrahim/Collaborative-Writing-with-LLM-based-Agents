# src/evaluation/evaluator.py
"""
Streamlined article evaluator that uses consolidated metric functions.

This evaluator delegates all metric calculations to the metrics module,
keeping the evaluator focused on orchestration and data handling.
"""

import logging
from typing import Dict

from src.evaluation.metrics import (
    METRIC_DESCRIPTIONS,
    STORM_METRICS,
    evaluate_article_metrics,
)
from src.utils.data_models import Article
from src.utils.freshwiki_loader import FreshWikiEntry

logger = logging.getLogger(__name__)


class ArticleEvaluator:
    """
    Evaluator using STORM paper metrics with consolidated metric functions.

    This class orchestrates the evaluation process but delegates all
    metric calculations to the centralized metrics module.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("ArticleEvaluator initialized with consolidated metrics")

    def evaluate_article(
        self, article: Article, reference: FreshWikiEntry
    ) -> Dict[str, float]:
        """
        Evaluate article using STORM paper metrics.

        Args:
            article: Generated article object with content
            reference: FreshWiki reference entry with content and outline

        Returns:
            Dictionary with all STORM metrics in percentage scale (0-100):
            - rouge_1, rouge_l (content overlap)
            - heading_soft_recall (topic coverage)
            - heading_entity_recall (entities in headings)
            - article_entity_recall (overall factual coverage)
        """
        try:
            # Delegate all metric calculation to the metrics module
            metrics = evaluate_article_metrics(
                article_content=article.content,
                reference_content=reference.reference_content,
                reference_headings=reference.reference_outline,
            )

            # Validate metrics are within expected ranges
            self.logger.debug(f"Evaluation completed: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            # Return zeros for all STORM metrics on failure
            return {metric: 0.0 for metric in STORM_METRICS}

    @staticmethod
    def get_metric_descriptions() -> Dict[str, str]:
        """Get descriptions of STORM metrics for documentation."""
        return METRIC_DESCRIPTIONS.copy()

    @staticmethod
    def get_supported_metrics() -> list:
        """Get list of supported metric names."""
        return STORM_METRICS.copy()

    def evaluate_batch(
        self, articles_and_references: list
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple articles in batch.

        Args:
            articles_and_references: List of (article, reference) tuples

        Returns:
            Dictionary mapping article identifiers to metric results
        """
        results = {}

        for i, (article, reference) in enumerate(articles_and_references):
            try:
                article_id = getattr(article, "id", f"article_{i}")
                results[article_id] = self.evaluate_article(article, reference)
                self.logger.debug(f"Evaluated article {article_id}")
            except Exception as e:
                self.logger.error(f"Failed to evaluate article {i}: {e}")
                results[f"article_{i}"] = {metric: 0.0 for metric in STORM_METRICS}

        self.logger.info(f"Batch evaluation completed for {len(results)} articles")
        return results

    def get_evaluation_summary(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Get a human-readable summary of evaluation results.

        Args:
            metrics: Dictionary of metric values

        Returns:
            Dictionary with formatted summary information
        """
        from src.evaluation.metrics import (
            calculate_composite_score,
            format_metrics_for_display,
        )

        formatted_metrics = format_metrics_for_display(metrics)
        composite_score = calculate_composite_score(metrics)

        summary = {
            "composite_score": f"{composite_score:.2f}%",
            "content_overlap": f"ROUGE-1: {formatted_metrics.get('rouge_1', '0.00%')}, ROUGE-L: {formatted_metrics.get('rouge_l', '0.00%')}",
            "heading_coverage": f"HSR: {formatted_metrics.get('heading_soft_recall', '0.00%')}, HER: {formatted_metrics.get('heading_entity_recall', '0.00%')}",
            "entity_coverage": f"AER: {formatted_metrics.get('article_entity_recall', '0.00%')}",
            "overall_quality": self._assess_overall_quality(composite_score),
        }

        return summary

    def evaluate_review_article(
        self,
        article: Article,
        reference: FreshWikiEntry,
        include_review_metrics: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate article with optional review-specific metrics.

        Args:
            article: Generated article object with content
            reference: FreshWiki reference entry
            include_review_metrics: Whether to include review-specific metrics

        Returns:
            Dictionary with STORM metrics and optionally review metrics
        """
        # Get standard STORM metrics
        metrics = self.evaluate_article(article, reference)

        if not include_review_metrics:
            return metrics

        # Add review-specific metrics if available in article metadata
        if "review_score" in article.metadata:
            metrics.update(
                {
                    "review_overall_score": article.metadata["review_score"] * 100,
                    "review_issues_count": article.metadata.get("review_issues", 0),
                    "review_recommendations_count": article.metadata.get(
                        "review_recommendations", 0
                    ),
                }
            )

            # Add category scores if available
            category_scores = article.metadata.get("category_scores", {})
            for category, score in category_scores.items():
                metrics[f"review_{category}_score"] = score * 100

        # Add collaboration metrics if available
        if "collaboration_iterations" in article.metadata:
            metrics.update(
                {
                    "collaboration_iterations": article.metadata[
                        "collaboration_iterations"
                    ],
                    "collaboration_initial_score": article.metadata.get(
                        "initial_score", 0
                    )
                    * 100,
                    "collaboration_final_score": article.metadata.get("final_score", 0)
                    * 100,
                    "collaboration_improvement": article.metadata.get(
                        "score_improvement", 0
                    )
                    * 100,
                    "collaboration_time": article.metadata.get("collaboration_time", 0),
                    "issues_resolved": article.metadata.get("issues_resolved", 0),
                    "recommendations_implemented": article.metadata.get(
                        "recommendations_implemented", 0
                    ),
                }
            )

        return metrics

    def get_extended_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all metrics including review-specific ones."""
        extended_descriptions = METRIC_DESCRIPTIONS.copy()

        # Add review metric descriptions
        extended_descriptions.update(
            {
                "review_overall_score": "Overall review quality score (0-100%)",
                "review_issues_count": "Number of issues identified by reviewer",
                "review_recommendations_count": "Number of recommendations provided",
                "review_factual_score": "Factual accuracy score from reviewer (0-100%)",
                "review_structural_score": "Structural quality score from reviewer (0-100%)",
                "review_clarity_score": "Clarity score from reviewer (0-100%)",
                "review_completeness_score": "Completeness score from reviewer (0-100%)",
                "review_style_score": "Style score from reviewer (0-100%)",
                "collaboration_iterations": "Number of collaboration iterations",
                "collaboration_initial_score": "Initial article score before collaboration (0-100%)",
                "collaboration_final_score": "Final article score after collaboration (0-100%)",
                "collaboration_improvement": "Score improvement from collaboration (0-100%)",
                "collaboration_time": "Total collaboration time in seconds",
                "issues_resolved": "Number of issues resolved during collaboration",
                "recommendations_implemented": "Number of recommendations implemented",
            }
        )

        return extended_descriptions

    def _assess_overall_quality(self, composite_score: float) -> str:
        """Provide qualitative assessment of overall quality."""
        if composite_score >= 80:
            return "Excellent"
        elif composite_score >= 70:
            return "Good"
        elif composite_score >= 60:
            return "Satisfactory"
        elif composite_score >= 50:
            return "Needs Improvement"
        else:
            return "Poor"
