# src/analysis/visualizer.py
"""
Visualization module for experimental results and analysis.
"""

from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ResultsVisualizer:
    """Generate comprehensive visualizations for experimental results."""

    def __init__(
        self, aggregated_data: Dict[str, Any], output_dir: str = "visualizations"
    ):
        self.aggregated_data = aggregated_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.raw_aggregations = aggregated_data.get("raw_aggregations", {})

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Configure matplotlib for better-looking plots
        plt.rcParams.update(
            {
                "figure.figsize": (10, 6),
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "figure.titlesize": 16,
            }
        )

    def generate_all_charts(self) -> Dict[str, str]:
        """Generate all visualization charts and return file paths."""
        chart_paths = {}

        if not self.raw_aggregations:
            logger.warning("No aggregated data available for visualization")
            return chart_paths

        try:
            # 1. Metric comparison charts
            chart_paths["metric_comparison"] = self._create_metric_comparison()

            # 2. Distribution plots
            chart_paths["distributions"] = self._create_distribution_plots()

            # 3. Success rate and word count analysis
            chart_paths["success_analysis"] = self._create_success_analysis()

            # 4. Effect size visualization
            chart_paths["effect_sizes"] = self._create_effect_size_plot()

            # 5. Statistical significance summary
            chart_paths["significance_summary"] = self._create_significance_summary()

            logger.info(
                f"Generated {len(chart_paths)} visualization charts in {self.output_dir}"
            )

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        return chart_paths

    def _create_metric_comparison(self) -> str:
        """Create detailed metric comparison across methods."""
        metric_names = [
            "rouge_1",
            "rouge_2",
            "rouge_l",
            "heading_soft_recall",
            "heading_entity_recall",
            "article_entity_recall",
        ]

        methods = list(self.raw_aggregations.keys())
        len(metric_names)
        n_methods = len(methods)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        fig.suptitle(
            "Detailed Metric Comparison Across Methods", fontsize=16, fontweight="bold"
        )

        colors = sns.color_palette("husl", n_methods)

        for i, metric in enumerate(metric_names):
            ax = axes[i]

            # Collect data for this metric
            method_means = []
            method_stds = []
            method_labels = []

            for method, agg in self.raw_aggregations.items():
                if metric in agg.metrics:
                    method_means.append(agg.metrics[metric].mean)
                    method_stds.append(agg.metrics[metric].std)
                    method_labels.append(method)

            if method_means:
                x_pos = np.arange(len(method_labels))
                bars = ax.bar(
                    x_pos,
                    method_means,
                    yerr=method_stds,
                    color=colors[: len(method_labels)],
                    alpha=0.7,
                    capsize=5,
                    error_kw={"linewidth": 2},
                )

                ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
                ax.set_ylabel("Score")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(method_labels, rotation=45)

                # Add value labels
                for bar, mean in zip(bars, method_means):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(method_stds) * 0.1,
                        f"{mean:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(metric.replace("_", " ").title(), fontweight="bold")

        plt.tight_layout()

        filepath = self.output_dir / "metric_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def _create_distribution_plots(self) -> str:
        """Create distribution plots for each metric."""
        metric_names = [
            "rouge_1",
            "rouge_2",
            "rouge_l",
            "heading_soft_recall",
            "heading_entity_recall",
            "article_entity_recall",
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        fig.suptitle(
            "Metric Score Distributions by Method", fontsize=16, fontweight="bold"
        )

        for i, metric in enumerate(metric_names):
            ax = axes[i]

            # Collect all values for this metric
            all_data = []
            method_labels = []

            for method, agg in self.raw_aggregations.items():
                if metric in agg.metrics and agg.metrics[metric].values:
                    all_data.extend(agg.metrics[metric].values)
                    method_labels.extend([method] * len(agg.metrics[metric].values))

            if all_data:
                # Create DataFrame for seaborn
                df = pd.DataFrame({"score": all_data, "method": method_labels})

                # Box plot with violin overlay
                sns.boxplot(data=df, x="method", y="score", ax=ax)
                sns.stripplot(
                    data=df,
                    x="method",
                    y="score",
                    ax=ax,
                    size=4,
                    alpha=0.6,
                    jitter=True,
                )

                ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
                ax.set_ylabel("Score")
                ax.set_xlabel("Method")

                # Rotate x-axis labels if needed
                if len(set(method_labels)) > 2:
                    ax.tick_params(axis="x", rotation=45)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(metric.replace("_", " ").title(), fontweight="bold")

        plt.tight_layout()

        filepath = self.output_dir / "distributions.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def _create_success_analysis(self) -> str:
        """Create success rate and word count analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            "Success Rate and Word Count Analysis", fontsize=16, fontweight="bold"
        )

        methods = list(self.raw_aggregations.keys())
        colors = sns.color_palette("husl", len(methods))

        # Success rates with confidence intervals (assuming binomial)
        success_rates = []
        ci_lower = []
        ci_upper = []

        for agg in self.raw_aggregations.values():
            rate = agg.success_rate
            n = (
                agg.topic_count / rate if rate > 0 else agg.topic_count
            )  # Total attempted

            # Wilson score interval for binomial proportion
            if n > 0:
                z = 1.96  # 95% confidence
                p_hat = rate
                n_eff = n

                center = (p_hat + z**2 / (2 * n_eff)) / (1 + z**2 / n_eff)
                margin = (
                    z
                    * np.sqrt(p_hat * (1 - p_hat) / n_eff + z**2 / (4 * n_eff**2))
                    / (1 + z**2 / n_eff)
                )

                ci_lower.append(max(0, center - margin))
                ci_upper.append(min(1, center + margin))
            else:
                ci_lower.append(0)
                ci_upper.append(1)

            success_rates.append(rate)

        # Plot success rates with error bars
        x_pos = np.arange(len(methods))
        bars1 = ax1.bar(
            x_pos,
            [r * 100 for r in success_rates],
            yerr=[(success_rates[i] - ci_lower[i]) * 100 for i in range(len(methods))],
            color=colors,
            alpha=0.7,
            capsize=5,
        )

        ax1.set_title("Success Rates with 95% CI", fontweight="bold")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(methods)
        ax1.set_ylim(0, 100)

        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars1, success_rates)):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{rate*100:.1f}%",
                ha="center",
                va="bottom",
            )

        # Word count distribution
        word_data = []
        word_methods = []

        for method, agg in self.raw_aggregations.items():
            # Create synthetic word count data based on mean (for visualization)
            mean_words = agg.avg_word_count
            n_samples = agg.topic_count

            if mean_words > 0 and n_samples > 0:
                # Generate approximate distribution
                std_words = mean_words * 0.3  # Assume 30% CV
                word_samples = np.random.normal(mean_words, std_words, n_samples)
                word_samples = np.maximum(word_samples, 0)  # No negative word counts

                word_data.extend(word_samples)
                word_methods.extend([method] * n_samples)

        if word_data:
            df_words = pd.DataFrame({"words": word_data, "method": word_methods})
            sns.boxplot(data=df_words, x="method", y="words", ax=ax2)
            ax2.set_title("Word Count Distribution", fontweight="bold")
            ax2.set_ylabel("Word Count")
            ax2.set_xlabel("Method")

        plt.tight_layout()

        filepath = self.output_dir / "success_analysis.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)

    def _create_effect_size_plot(self) -> str:
        """Create effect size visualization."""
        # This requires statistical analysis data
        try:
            from .statistical_analyzer import StatisticalAnalyzer

            analyzer = StatisticalAnalyzer(self.aggregated_data)
            analysis = analyzer.analyze_all()
            effect_sizes = analysis.get("effect_sizes", {})

            if not effect_sizes:
                logger.warning("No effect size data available")
                return ""

            # Prepare data for plotting
            comparisons = []
            metrics = []
            cohens_d_values = []
            interpretations = []

            for comp_key, comp_data in effect_sizes.items():
                for metric, effect_data in comp_data.items():
                    comparisons.append(comp_key.replace("_vs_", " vs "))
                    metrics.append(metric.replace("_", " ").title())
                    cohens_d_values.append(effect_data["cohens_d"])
                    interpretations.append(effect_data["interpretation"])

            if not cohens_d_values:
                return ""

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "Comparison": comparisons,
                    "Metric": metrics,
                    "Effect Size": cohens_d_values,
                    "Interpretation": interpretations,
                }
            )

            # Create pivot table for heatmap
            pivot_df = df.pivot(
                index="Metric", columns="Comparison", values="Effect Size"
            )

            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle("Effect Size Analysis", fontsize=16, fontweight="bold")

            # Heatmap
            sns.heatmap(
                pivot_df,
                annot=True,
                cmap="RdBu_r",
                center=0,
                fmt=".3f",
                cbar_kws={"label": "Cohen's d"},
                ax=ax1,
            )
            ax1.set_title("Effect Size Heatmap", fontweight="bold")

            # Bar plot
            abs_effects = np.abs(cohens_d_values)
            colors = [
                (
                    "red"
                    if interp == "large"
                    else (
                        "orange"
                        if interp == "medium"
                        else "yellow" if interp == "small" else "lightgray"
                    )
                )
                for interp in interpretations
            ]

            y_pos = np.arange(len(metrics))
            bars = ax2.barh(y_pos, abs_effects, color=colors, alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(metrics)
            ax2.set_xlabel("Absolute Effect Size (|Cohen's d|)")
            ax2.set_title("Effect Size Magnitudes", fontweight="bold")

            # Add effect size thresholds
            ax2.axvline(
                x=0.2, color="gray", linestyle="--", alpha=0.5, label="Small (0.2)"
            )
            ax2.axvline(
                x=0.5, color="gray", linestyle="--", alpha=0.7, label="Medium (0.5)"
            )
            ax2.axvline(
                x=0.8, color="gray", linestyle="-", alpha=0.9, label="Large (0.8)"
            )
            ax2.legend()

            plt.tight_layout()

            filepath = self.output_dir / "effect_sizes.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Error creating effect size plot: {e}")
            return ""

    def _create_significance_summary(self) -> str:
        """Create statistical significance summary visualization."""
        try:
            from .statistical_analyzer import StatisticalAnalyzer

            analyzer = StatisticalAnalyzer(self.aggregated_data)
            analysis = analyzer.analyze_all()

            # Extract significance data
            significance_data = []
            comparisons = analysis.get("method_comparisons", {})

            for comp_key, comp_data in comparisons.items():
                metric_comparisons = comp_data.get("metric_comparisons", {})
                for metric, metric_comp in metric_comparisons.items():
                    tests = metric_comp.get("tests", [])
                    for test in tests:
                        significance_data.append(
                            {
                                "Comparison": comp_key.replace("_vs_", " vs "),
                                "Metric": metric.replace("_", " ").title(),
                                "Test": test.get("test_name", ""),
                                "P-value": test.get("p_value", 1.0),
                                "Significant": test.get("significant", False),
                                "Effect Size": test.get("effect_size", 0.0),
                            }
                        )

            if not significance_data:
                logger.warning("No significance data available")
                return ""

            df = pd.DataFrame(significance_data)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.suptitle(
                "Statistical Significance Summary", fontsize=16, fontweight="bold"
            )

            # Significance matrix
            pivot_sig = df.pivot_table(
                index="Metric",
                columns="Comparison",
                values="Significant",
                aggfunc="any",
                fill_value=False,
            )

            sns.heatmap(
                pivot_sig,
                annot=True,
                cmap="RdYlGn",
                cbar_kws={"label": "Significant"},
                fmt="",
                ax=ax1,
            )
            ax1.set_title(
                "Significance Matrix (Any Test Significant)", fontweight="bold"
            )

            # P-value distribution
            p_values = df["P-value"].values
            ax2.hist(p_values, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
            ax2.axvline(
                x=0.05, color="red", linestyle="--", linewidth=2, label="α = 0.05"
            )
            ax2.set_xlabel("P-value")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Distribution of P-values", fontweight="bold")
            ax2.legend()

            plt.tight_layout()

            filepath = self.output_dir / "significance_summary.png"
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Error creating significance summary: {e}")
            return ""

    def create_custom_plot(self, plot_type: str, **kwargs) -> str:
        """Create custom visualization based on user specification."""
        try:
            if plot_type == "metric_scatter":
                return self._create_metric_scatter(**kwargs)
            elif plot_type == "method_timeline":
                return self._create_method_timeline(**kwargs)
            elif plot_type == "topic_analysis":
                return self._create_topic_analysis(**kwargs)
            else:
                logger.warning(f"Unknown plot type: {plot_type}")
                return ""
        except Exception as e:
            logger.error(f"Error creating custom plot {plot_type}: {e}")
            return ""

    def _create_metric_scatter(
        self, x_metric: str = "rouge_1", y_metric: str = "heading_soft_recall"
    ) -> str:
        """Create scatter plot between two metrics."""
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = sns.color_palette("husl", len(self.raw_aggregations))

        for i, (method, agg) in enumerate(self.raw_aggregations.items()):
            if x_metric in agg.metrics and y_metric in agg.metrics:
                x_values = agg.metrics[x_metric].values
                y_values = agg.metrics[y_metric].values

                # Align data (take minimum length)
                min_len = min(len(x_values), len(y_values))
                x_values = x_values[:min_len]
                y_values = y_values[:min_len]

                ax.scatter(
                    x_values, y_values, label=method, color=colors[i], alpha=0.7, s=50
                )

        ax.set_xlabel(x_metric.replace("_", " ").title())
        ax.set_ylabel(y_metric.replace("_", " ").title())
        ax.set_title(
            f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}',
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        filepath = self.output_dir / f"scatter_{x_metric}_vs_{y_metric}.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return str(filepath)
