# src/analysis/aggregator.py
"""
Metrics aggregation module for baseline comparison analysis.
"""

from collections import defaultdict

import logging
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Any, Dict, List

from .data_loader import TopicResult

logger = logging.getLogger(__name__)


@dataclass
class MetricStats(BaseModel):
    """Statistical summary for a single metric."""

    metric_name: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    count: int
    values: List[float]


@dataclass
class MethodAggregation(BaseModel):
    """Aggregated metrics for a single method."""

    method: str
    topic_count: int
    success_rate: float
    avg_word_count: float
    metrics: Dict[str, MetricStats]

    def get_metric_means(self) -> Dict[str, float]:
        """Get mean values for all metrics."""
        return {name: stats.mean for name, stats in self.metrics.items()}


class MetricsAggregator(BaseModel):
    """Aggregate and summarize evaluation metrics across baselines."""

    from typing import ClassVar

    STORM_METRICS: ClassVar[List[str]] = [
        "rouge_1",
        "rouge_l",
        "heading_soft_recall",
        "heading_entity_recall",
        "article_entity_recall",
    ]

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.results: List[TopicResult] = []

    def set_results(self, results: List[TopicResult]):
        """Set the topic results to aggregate."""
        self.results = results
        logger.info(f"Set {len(results)} results for aggregation")

    def aggregate_by_method(self) -> Dict[str, MethodAggregation]:
        """Aggregate metrics by method (direct vs storm)."""
        if not self.results:
            raise ValueError("No results available. Call set_results() first.")

        method_groups = defaultdict(list)
        for result in self.results:
            if result.success and result.evaluation:
                method_groups[result.method].append(result)

        aggregations = {}
        for method, method_results in method_groups.items():
            aggregations[method] = self._aggregate_method_results(
                method, method_results
            )

        logger.info(f"Aggregated metrics for methods: {list(aggregations.keys())}")
        return aggregations

    def _aggregate_method_results(
        self, method: str, results: List[TopicResult]
    ) -> MethodAggregation:
        """Aggregate results for a single method."""
        if not results:
            raise ValueError(f"No results for method {method}")

        # Calculate basic stats
        total_results = len([r for r in self.results if r.method == method])
        successful_results = len(results)
        success_rate = successful_results / total_results if total_results > 0 else 0

        word_counts = [r.word_count for r in results if r.word_count > 0]
        avg_word_count = np.mean(word_counts) if word_counts else 0

        # Aggregate metrics
        metric_stats = {}
        for metric in self.STORM_METRICS:
            values = []
            for result in results:
                if result.evaluation and metric in result.evaluation:
                    values.append(result.evaluation[metric])

            if values:
                metric_stats[metric] = MetricStats(
                    metric_name=metric,
                    mean=float(np.mean(values)),
                    median=float(np.median(values)),
                    std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    min=float(np.min(values)),
                    max=float(np.max(values)),
                    count=len(values),
                    values=values,
                )
            else:
                logger.warning(
                    f"No values found for metric {metric} in method {method}"
                )

        return MethodAggregation(
            method=method,
            topic_count=successful_results,
            success_rate=success_rate,
            avg_word_count=avg_word_count,
            metrics=metric_stats,
        )

    def compare_methods(self, method1: str, method2: str) -> Dict[str, Any]:
        """Compare two methods across all metrics."""
        aggregations = self.aggregate_by_method()

        if method1 not in aggregations or method2 not in aggregations:
            raise ValueError(f"Methods {method1} or {method2} not found in results")

        agg1 = aggregations[method1]
        agg2 = aggregations[method2]

        comparison = {
            "method1": method1,
            "method2": method2,
            "sample_sizes": {method1: agg1.topic_count, method2: agg2.topic_count},
            "success_rates": {method1: agg1.success_rate, method2: agg2.success_rate},
            "word_counts": {method1: agg1.avg_word_count, method2: agg2.avg_word_count},
            "metric_comparisons": {},
        }

        # Compare each metric
        for metric in self.STORM_METRICS:
            if metric in agg1.metrics and metric in agg2.metrics:
                stats1 = agg1.metrics[metric]
                stats2 = agg2.metrics[metric]

                comparison["metric_comparisons"][metric] = {
                    "means": {method1: stats1.mean, method2: stats2.mean},
                    "medians": {method1: stats1.median, method2: stats2.median},
                    "stds": {method1: stats1.std, method2: stats2.std},
                    "difference": stats2.mean - stats1.mean,
                    "percent_change": (
                        ((stats2.mean - stats1.mean) / stats1.mean * 100)
                        if stats1.mean > 0
                        else float("inf")
                    ),
                }

        return comparison

    def aggregate_all_metrics(self) -> Dict[str, Any]:
        """Aggregate all metrics and return comprehensive summary."""
        from .data_loader import ResultsLoader

        # If results not set, extract from data
        if not self.results:
            loader = ResultsLoader("")
            loader.data = self.data
            self.results = loader.get_successful_results()

        method_aggregations = self.aggregate_by_method()

        # Overall statistics
        total_topics = len(set(r.topic for r in self.results))
        total_successful = len(self.results)

        # Cross-method comparison if we have multiple methods
        comparisons = {}
        methods = list(method_aggregations.keys())
        if len(methods) >= 2:
            for i, method1 in enumerate(methods):
                for method2 in methods[i + 1 :]:
                    comp_key = f"{method1}_vs_{method2}"
                    comparisons[comp_key] = self.compare_methods(method1, method2)

        return {
            "summary": {
                "total_topics": total_topics,
                "total_successful": total_successful,
                "methods": methods,
            },
            "method_aggregations": {
                method: {
                    "basic_stats": {
                        "topic_count": agg.topic_count,
                        "success_rate": agg.success_rate,
                        "avg_word_count": agg.avg_word_count,
                    },
                    "metrics": {
                        name: stats.model_dump() for name, stats in agg.metrics.items()
                    },
                }
                for method, agg in method_aggregations.items()
            },
            "comparisons": comparisons,
            "raw_aggregations": method_aggregations,  # For use by other modules
        }

    def get_metric_distributions(self) -> Dict[str, Dict[str, List[float]]]:
        """Get value distributions for each metric by method."""
        if not self.results:
            raise ValueError("No results available")

        distributions = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            if result.success and result.evaluation:
                for metric, value in result.evaluation.items():
                    if metric in self.STORM_METRICS:
                        distributions[metric][result.method].append(value)

        return dict(distributions)

    def get_topic_level_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get topic-by-topic comparison for paired analysis."""
        if not self.results:
            raise ValueError("No results available")

        # Group by topic
        topic_groups = defaultdict(dict)
        for result in self.results:
            if result.success and result.evaluation:
                topic_groups[result.topic][result.method] = result

        # Find topics with results for multiple methods
        paired_topics = {}
        for topic, methods in topic_groups.items():
            if len(methods) >= 2:
                paired_topics[topic] = {}
                for method, result in methods.items():
                    paired_topics[topic][method] = {
                        "word_count": result.word_count,
                        "metrics": result.evaluation,
                    }

        logger.info(f"Found {len(paired_topics)} topics with multiple method results")
        return paired_topics
