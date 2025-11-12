from pathlib import Path

import json
from logging import getLogger
from typing import Dict, Optional

logger = getLogger(__name__)


def extract_metrics_from_results(results_file: Path) -> Optional[Dict]:
    """Extract metrics from results.json file."""
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return None

    try:
        with open(results_file) as f:
            data = json.load(f)

        # Get evaluation aggregate (should have method as key)
        agg = data.get("evaluation_aggregate", {})
        if not agg:
            logger.warning(f"No evaluation_aggregate in {results_file}")
            return None

        # Get first method's metrics
        method_data = next(iter(agg.values())) if agg else {}

        metrics = {
            "rouge_1": method_data.get("rouge_1"),
            "rouge_l": method_data.get("rouge_l"),
            "heading_soft_recall": method_data.get("heading_soft_recall"),
            "heading_entity_recall": method_data.get("heading_entity_recall"),
            "article_entity_recall": method_data.get("article_entity_recall"),
        }

        # LLM judge metrics
        llm_judge = method_data.get("llm_judge", {})
        if llm_judge:
            judge_scores = [
                llm_judge.get("interest_level", 0),
                llm_judge.get("coherence_organization", 0),
                llm_judge.get("relevance_focus", 0),
                llm_judge.get("broad_coverage", 0),
            ]
            if any(judge_scores):
                metrics["llm_judge_avg"] = sum(judge_scores) / len(
                    [s for s in judge_scores if s > 0]
                )

        return metrics

    except Exception as e:
        logger.error(f"Failed to extract metrics from {results_file}: {e}")
        return None


def calculate_composite_score(metrics: Dict) -> float:
    """
    Calculate composite score from metrics.
    Weights are based on importance for FreshWiki evaluation.
    """
    weights = {
        "rouge_1": 0.25,
        "rouge_l": 0.15,
        "heading_soft_recall": 0.10,
        "heading_entity_recall": 0.10,
        "article_entity_recall": 0.15,
        "llm_judge_avg": 0.25,
    }

    score = 0.0
    total_weight = 0.0

    for metric, weight in weights.items():
        value = metrics.get(metric)
        if value is not None:
            # Normalize to 0-100 scale
            if "rouge" in metric or "recall" in metric:
                normalized = value  # Already 0-100
            elif "llm_judge" in metric:
                normalized = value * 20  # 0-5 -> 0-100
            else:
                normalized = value

            score += normalized * weight
            total_weight += weight

    if total_weight > 0:
        return score / total_weight
    return 0.0
