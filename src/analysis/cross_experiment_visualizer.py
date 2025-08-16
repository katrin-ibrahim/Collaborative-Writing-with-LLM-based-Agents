# src/analysis/cross_experiment_visualizer.py
"""
Cross-experiment comparison visualizations.
Compare the same methods across different experimental runs.
"""

from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict, List

from .data_loader import ResultsLoader

logger = logging.getLogger(__name__)


class CrossExperimentVisualizer:
    """Generate visualizations comparing methods across multiple experiments."""

    def __init__(
        self,
        experiment_paths: List[str],
        output_dir: str = "results/cross_experiment_analysis",
    ):
        self.experiment_paths = [Path(p) for p in experiment_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Standard method ordering
        self.method_order = ["direct", "rag", "storm"]

        # Load all experiments
        self.experiments = {}
        self._load_experiments()

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        plt.rcParams.update(
            {
                "figure.figsize": (12, 8),
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "figure.titlesize": 16,
            }
        )

    def _load_experiments(self):
        """Load data from all experiment directories."""
        for exp_path in self.experiment_paths:
            results_file = exp_path / "results.json"
            if results_file.exists():
                try:
                    loader = ResultsLoader(str(results_file))
                    data = loader.load_and_validate()

                    # Use directory name as experiment identifier
                    exp_name = exp_path.name
                    self.experiments[exp_name] = {
                        "data": data,
                        "metadata": loader.metadata,
                        "path": exp_path,
                    }
                    logger.info(f"Loaded experiment: {exp_name}")
                except Exception as e:
                    logger.warning(f"Failed to load experiment from {exp_path}: {e}")
            else:
                logger.warning(f"No results.json found in {exp_path}")

    def create_method_comparison_chart(self, method: str) -> str:
        """Create comparison chart for a specific method across experiments."""
        if not self.experiments:
            logger.error("No experiments loaded")
            return ""

        # Collect data for the specified method
        experiment_data = []

        for exp_name, exp_info in self.experiments.items():
            results = exp_info["data"].get("results", {})

            # Extract metrics for this method across all topics
            for topic, topic_data in results.items():
                if method in topic_data and "evaluation" in topic_data[method]:
                    metrics = topic_data[method]["evaluation"]

                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            experiment_data.append(
                                {
                                    "experiment": exp_name,
                                    "topic": topic,
                                    "method": method,
                                    "metric": metric_name,
                                    "value": metric_value,
                                }
                            )

        if not experiment_data:
            logger.warning(f"No data found for method: {method}")
            return ""

        df = pd.DataFrame(experiment_data)

        # Create visualization
        metrics = df["metric"].unique()
        n_metrics = len(metrics)

        # Determine grid size
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_metrics > 1 else [axes]
        else:
            axes = axes.flatten()

        fig.suptitle(
            f"{method.upper()} Method Performance Across Experiments",
            fontsize=16,
            fontweight="bold",
        )

        for i, metric in enumerate(metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue

            metric_data = df[df["metric"] == metric]

            # Box plot comparing experiments
            sns.boxplot(data=metric_data, x="experiment", y="value", ax=ax)
            sns.stripplot(
                data=metric_data,
                x="experiment",
                y="value",
                ax=ax,
                size=4,
                alpha=0.7,
                jitter=True,
                color="black",
            )

            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel("Experiment")
            ax.set_ylabel("Score")
            ax.tick_params(axis="x", rotation=45)

        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        filepath = self.output_dir / f"{method}_cross_experiment_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created {method} comparison chart: {filepath}")
        return str(filepath)

    def create_all_methods_summary(self) -> str:
        """Create summary comparison of all methods across experiments."""
        if not self.experiments:
            logger.error("No experiments loaded")
            return ""

        # Collect aggregated data
        summary_data = []

        for exp_name, exp_info in self.experiments.items():
            results = exp_info["data"].get("results", {})

            # Aggregate by method for this experiment
            method_aggregates = {}

            for topic, topic_data in results.items():
                for method, method_data in topic_data.items():
                    if "evaluation" in method_data:
                        if method not in method_aggregates:
                            method_aggregates[method] = {}

                        metrics = method_data["evaluation"]
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                if metric_name not in method_aggregates[method]:
                                    method_aggregates[method][metric_name] = []
                                method_aggregates[method][metric_name].append(
                                    metric_value
                                )

            # Calculate means for each method/metric
            for method, metrics in method_aggregates.items():
                for metric_name, values in metrics.items():
                    summary_data.append(
                        {
                            "experiment": exp_name,
                            "method": method,
                            "metric": metric_name,
                            "mean_score": np.mean(values),
                            "std_score": np.std(values),
                            "count": len(values),
                        }
                    )

        if not summary_data:
            logger.warning("No summary data found")
            return ""

        df = pd.DataFrame(summary_data)

        # Create visualization
        metrics = df["metric"].unique()
        n_metrics = len(metrics)

        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_metrics > 1 else [axes]
        else:
            axes = axes.flatten()

        fig.suptitle(
            "Method Performance Summary Across All Experiments",
            fontsize=16,
            fontweight="bold",
        )

        # Get ordered methods
        methods = [m for m in self.method_order if m in df["method"].unique()]
        for m in df["method"].unique():
            if m not in methods:
                methods.append(m)

        for i, metric in enumerate(metrics):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue

            metric_data = df[df["metric"] == metric]

            # Pivot for easier plotting
            pivot_data = metric_data.pivot(
                index="experiment", columns="method", values="mean_score"
            )
            pivot_std = metric_data.pivot(
                index="experiment", columns="method", values="std_score"
            )

            # Reorder columns to match method order
            available_methods = [m for m in methods if m in pivot_data.columns]
            pivot_data = pivot_data[available_methods]
            pivot_std = pivot_std[available_methods]

            # Bar plot with error bars
            pivot_data.plot(
                kind="bar", ax=ax, yerr=pivot_std, capsize=4, width=0.8, alpha=0.8
            )

            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel("Experiment")
            ax.set_ylabel("Mean Score")
            ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=45)

        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        filepath = self.output_dir / "all_methods_cross_experiment_summary.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Created summary comparison chart: {filepath}")
        return str(filepath)

    def generate_all_charts(self) -> Dict[str, str]:
        """Generate all cross-experiment comparison charts."""
        chart_paths = {}

        if not self.experiments:
            logger.error("No experiments loaded")
            return chart_paths

        # Get all available methods
        all_methods = set()
        for exp_info in self.experiments.values():
            results = exp_info["data"].get("results", {})
            for topic_data in results.values():
                all_methods.update(topic_data.keys())

        # Create individual method comparisons
        for method in all_methods:
            if method in self.method_order or method not in ["summary", "metadata"]:
                chart_path = self.create_method_comparison_chart(method)
                if chart_path:
                    chart_paths[f"{method}_comparison"] = chart_path

        # Create summary comparison
        summary_path = self.create_all_methods_summary()
        if summary_path:
            chart_paths["summary_comparison"] = summary_path

        return chart_paths

    def get_experiment_info(self) -> Dict[str, Any]:
        """Get information about loaded experiments."""
        info = {
            "total_experiments": len(self.experiments),
            "experiment_names": list(self.experiments.keys()),
            "experiment_details": {},
        }

        for exp_name, exp_info in self.experiments.items():
            metadata = exp_info.get("metadata")
            results = exp_info["data"].get("results", {})

            methods = set()
            total_topics = len(results)

            for topic_data in results.values():
                methods.update(topic_data.keys())

            info["experiment_details"][exp_name] = {
                "path": str(exp_info["path"]),
                "total_topics": total_topics,
                "methods": list(methods),
                "timestamp": metadata.timestamp if metadata else "unknown",
            }

        return info
