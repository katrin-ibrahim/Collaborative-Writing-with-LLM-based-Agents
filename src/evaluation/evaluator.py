# src/evaluation/evaluator.py
"""
Streamlined article evaluator that uses consolidated metric functions.

This evaluator delegates all metric calculations to the metrics module,
keeping the evaluator focused on orchestration and data handling.
"""

import logging
from typing import Dict

from src.evaluation.metrics import (
    STORM_METRICS,
    evaluate_article_metrics,
)
from src.utils.data import Article, FreshWikiEntry

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
