# src/analysis/__init__.py
"""
Modular analysis framework for AI Writer Agent baseline experiments.

This package provides comprehensive tools for analyzing results.json files
from baseline experiments, including statistical analysis, visualization,
and insight extraction.
"""

from .aggregator import MetricsAggregator
from .data_loader import ResultsLoader
from .statistical_analyzer import StatisticalAnalyzer
from .visualizer import ResultsVisualizer

__all__ = [
    "ResultsLoader",
    "MetricsAggregator",
    "StatisticalAnalyzer",
    "ResultsVisualizer",
]

# Version info
__version__ = "1.0.0"
__author__ = "AI Research Team"


# Helper function to convert aggregated data to JSON-serializable format
def _convert_aggregated_to_json(aggregated_data):
    """Convert aggregated data to JSON-serializable format."""
    json_data = {"summary": aggregated_data.get("summary", {}), "raw_aggregations": {}}

    # Convert raw aggregations
    raw_aggs = aggregated_data.get("raw_aggregations", {})
    for method, agg in raw_aggs.items():
        json_data["raw_aggregations"][method] = {
            "topic_count": agg.topic_count,
            "success_rate": agg.success_rate,
            "avg_word_count": agg.avg_word_count,
            "metrics": {},
        }

        # Convert metrics
        for metric_name, metric_stats in agg.metrics.items():
            json_data["raw_aggregations"][method]["metrics"][metric_name] = {
                "mean": metric_stats.mean,
                "std": metric_stats.std,
                "min": metric_stats.min,
                "max": metric_stats.max,
                "count": metric_stats.count,
                # Removed "values" to keep file size manageable
            }

    return json_data


# Main analysis pipeline function
def analyze_results(results_dir: str):
    """
    Complete analysis pipeline for baseline experiment results.

    Args:
        results_dir: Path to results directory (must contain results.json)

    Returns:
        Dict containing all analysis results and file paths
    """
    from pathlib import Path

    import os

    # Handle input path - must be a directory
    input_path = Path(results_dir)

    if not input_path.is_dir():
        raise ValueError(f"Input path must be a directory: {results_dir}")

    # Always append results.json to the directory
    results_file_path = input_path / "results.json"

    # Output directory is analysis_output inside the results directory
    output_dir = input_path / "analysis_output"

    # Convert to string for compatibility
    results_file_path = str(results_file_path)
    output_dir = str(output_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    loader = ResultsLoader(results_file_path)
    data = loader.load_and_validate()

    # Aggregate metrics
    aggregator = MetricsAggregator(data)
    aggregated = aggregator.aggregate_all_metrics()

    # Save aggregated results to file
    import json

    aggregated_file_path = Path(output_dir) / "aggregated_metrics.json"
    with open(aggregated_file_path, "w") as f:
        # Convert aggregated data to JSON-serializable format
        json_data = _convert_aggregated_to_json(aggregated)
        json.dump(json_data, f, indent=2)

    # Statistical analysis
    analyzer = StatisticalAnalyzer(aggregated)
    stats = analyzer.analyze_all()

    # Generate visualizations
    visualizer = ResultsVisualizer(aggregated, output_dir)
    charts = visualizer.generate_all_charts()

    return {
        "data": data,
        "aggregated": aggregated,
        "statistics": stats,
        "charts": charts,
        "aggregated_file": str(aggregated_file_path),
    }
