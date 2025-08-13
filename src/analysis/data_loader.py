# src/analysis/data_loader.py
"""
Data loading and validation module for experiment results.
"""

from pathlib import Path

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata from experiment configuration."""

    timestamp: str
    ollama_host: str
    methods: List[str]
    num_topics: int
    models: Dict[str, str]
    total_time: float
    topics_processed: int


@dataclass
class TopicResult:
    """Results for a single topic and method."""

    topic: str
    method: str
    success: bool
    word_count: int = 0
    evaluation: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class ResultsLoader:
    """Load and validate experiment results from JSON files."""

    REQUIRED_METRICS = [
        "rouge_1",
        "rouge_l",
        "heading_soft_recall",
        "heading_entity_recall",
        "article_entity_recall",
    ]

    def __init__(self, results_file_path: str):
        self.results_file = Path(results_file_path)
        self.data = None
        self.metadata = None

    def load_and_validate(self) -> Dict[str, Any]:
        """Load results file and validate structure."""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        try:
            with open(self.results_file, "r") as f:
                self.data = json.load(f)
            logger.info(f"Loaded results from {self.results_file}")

            # Validate structure
            self._validate_structure()

            # Extract metadata
            self.metadata = self._extract_metadata()

            return self.data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in results file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load results: {e}")

    def _validate_structure(self):
        """Validate that results have expected structure."""
        if not isinstance(self.data, dict):
            raise ValueError("Results must be a dictionary")

        # Try new structure first (summary + results), fallback to old structure
        if "summary" in self.data and "results" in self.data:
            # New structure - results from evaluation
            required_keys = ["summary", "results"]
            missing_keys = [key for key in required_keys if key not in self.data]
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")
        else:
            # Old structure - original format
            required_keys = ["timestamp", "configuration", "results"]
            missing_keys = [key for key in required_keys if key not in self.data]
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")

        # Validate results structure
        results = self.data.get("results", {})
        if not isinstance(results, dict):
            raise ValueError("Results section must be a dictionary")

        # Check for valid method data and filter out failed entries
        valid_methods = set()
        filtered_results = {}

        for topic, topic_data in results.items():
            if not isinstance(topic_data, dict):
                continue

            filtered_topic_data = {}
            for method, method_data in topic_data.items():
                if method in ["direct", "storm", "rag"]:
                    # Only include successful results with actual article content
                    if (
                        isinstance(method_data, dict)
                        and method_data.get("success", False)
                        and "evaluation" in method_data
                        and method_data.get("error") != "Article file not found"
                    ):

                        filtered_topic_data[method] = method_data
                        valid_methods.add(method)
                    else:
                        logger.info(f"Filtering out failed result: {topic}/{method}")

            # Only include topics that have at least one successful method
            if filtered_topic_data:
                filtered_results[topic] = filtered_topic_data

        # Update the data with filtered results
        self.data["results"] = filtered_results

        if not valid_methods:
            raise ValueError(
                "No valid method results found (expected 'direct', 'storm', or 'rag')"
            )

        logger.info(f"Validation passed. Found methods: {sorted(valid_methods)}")
        logger.info(
            f"Filtered to {len(filtered_results)} topics with successful results"
        )

    def _extract_metadata(self) -> ExperimentMetadata:
        """Extract experiment metadata."""
        # Handle both old and new structure
        if "summary" in self.data:
            # New structure from evaluation results
            summary = self.data["summary"]
            timestamp = summary.get(
                "timestamp", self.data.get("evaluation_timestamp", "")
            )

            # Extract methods from summary
            methods = []
            if "methods" in summary:
                methods = list(summary["methods"].keys())

            return ExperimentMetadata(
                timestamp=timestamp,
                ollama_host="unknown",  # Not available in new structure
                methods=methods,
                num_topics=summary.get("topics_processed", 0),
                models={},  # Extract from methods if available
                total_time=summary.get("total_time", 0.0),
                topics_processed=summary.get("topics_processed", 0),
            )
        else:
            # Old structure
            config = self.data["configuration"]
            summary = self.data.get("summary", {})

            return ExperimentMetadata(
                timestamp=self.data["timestamp"],
                ollama_host=config.get("ollama_host", ""),
                methods=config.get("methods", []),
                num_topics=config.get("num_topics", 0),
                models=config.get("models", {}),
                total_time=summary.get("total_time", 0.0),
                topics_processed=summary.get("topics_processed", 0),
            )

    def get_topic_results(self) -> List[TopicResult]:
        """Extract structured topic results."""
        if not self.data:
            raise RuntimeError("No data loaded. Call load_and_validate() first.")

        topic_results = []
        results = self.data["results"]

        for topic, topic_data in results.items():
            for method, method_data in topic_data.items():
                if method not in ["direct", "storm", "rag"]:
                    continue

                # Extract evaluation metrics if available
                evaluation = None
                if "metrics" in method_data:
                    eval_data = method_data["metrics"]
                    if isinstance(eval_data, dict):
                        # Validate all required metrics are present
                        if all(metric in eval_data for metric in self.REQUIRED_METRICS):
                            evaluation = eval_data
                        else:
                            missing = [
                                m for m in self.REQUIRED_METRICS if m not in eval_data
                            ]
                            logger.warning(
                                f"Missing metrics for {topic}/{method}: {missing}"
                            )
                elif "evaluation" in method_data:
                    eval_data = method_data["evaluation"]
                    if isinstance(eval_data, dict):
                        # Validate all required metrics are present
                        if all(metric in eval_data for metric in self.REQUIRED_METRICS):
                            evaluation = eval_data
                        else:
                            missing = [
                                m for m in self.REQUIRED_METRICS if m not in eval_data
                            ]
                            logger.warning(
                                f"Missing metrics for {topic}/{method}: {missing}"
                            )

                topic_result = TopicResult(
                    topic=topic,
                    method=method,
                    success=method_data.get("success", False),
                    word_count=method_data.get("word_count", 0),
                    evaluation=evaluation,
                    error=method_data.get("error")
                    or method_data.get("evaluation_error"),
                )

                topic_results.append(topic_result)

        logger.info(f"Extracted {len(topic_results)} topic results")
        return topic_results

    def get_successful_results(self) -> List[TopicResult]:
        """Get only successful results with valid evaluation metrics."""
        all_results = self.get_topic_results()
        successful = [r for r in all_results if r.success and r.evaluation is not None]

        logger.info(f"Found {len(successful)} successful results with evaluations")
        return successful

    def get_methods(self) -> List[str]:
        """Get list of methods found in results."""
        if not self.data:
            return []
        return list(
            set(
                method
                for topic_data in self.data["results"].values()
                for method in topic_data.keys()
                if method in ["direct", "storm", "rag"]
            )
        )

    def get_topics(self) -> List[str]:
        """Get list of topics in results."""
        if not self.data:
            return []
        return list(self.data["results"].keys())

    def summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the loaded data."""
        if not self.data:
            return {}

        results = self.get_topic_results()
        successful = self.get_successful_results()

        methods = self.get_methods()
        method_stats = {}

        for method in methods:
            method_results = [r for r in results if r.method == method]
            method_successful = [r for r in successful if r.method == method]

            method_stats[method] = {
                "total": len(method_results),
                "successful": len(method_successful),
                "success_rate": (
                    len(method_successful) / len(method_results)
                    if method_results
                    else 0
                ),
                "avg_word_count": (
                    sum(r.word_count for r in method_successful)
                    / len(method_successful)
                    if method_successful
                    else 0
                ),
            }

        return {
            "metadata": self.metadata.__dict__ if self.metadata else {},
            "total_topics": len(self.get_topics()),
            "total_results": len(results),
            "successful_results": len(successful),
            "methods": methods,
            "method_stats": method_stats,
        }
