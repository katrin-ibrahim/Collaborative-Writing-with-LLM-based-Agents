"""
Visualization tools for collaboration and Theory of Mind metrics.
"""

from pathlib import Path

import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CollaborationMetricsVisualizer:
    """Create visualizations for collaboration metrics."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer with output directory."""
        self.output_dir = (
            Path(output_dir) if output_dir else Path("results/visualizations")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        plt.rcParams.update(
            {
                "figure.figsize": (10, 6),
                "font.size": 13,
                "axes.titlesize": 15,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
            }
        )

    def plot_feedback_resolution(
        self, metrics: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot feedback resolution metrics over iterations.

        Args:
            metrics: Collaboration metrics dictionary
            save_path: Optional custom save path

        Returns:
            Path to saved plot
        """
        if not save_path:
            save_path = self.output_dir / "feedback_resolution.png"

        logger.info("Plotting feedback resolution metrics")
        logger.debug(f"Metrics data: {metrics}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Feedback counts by iteration
        feedback_by_iter = metrics.get("feedback_by_iteration", {})
        if feedback_by_iter:
            logger.info(f"Feedback by iteration: {feedback_by_iter}")
            iterations = sorted([int(k) for k in feedback_by_iter.keys()])
            counts = [feedback_by_iter[str(i)] for i in iterations]

            ax1.bar(
                iterations,
                counts,
                alpha=0.7,
                color="#4575b4",
                edgecolor="black",
                linewidth=1,
            )
            ax1.set_xlabel("Iteration", fontsize=14)
            ax1.set_ylabel("Number of Feedback Items", fontsize=14)
            ax1.set_title(
                "Feedback Items per Iteration", fontsize=15, fontweight="bold"
            )
            ax1.grid(axis="y", alpha=0.3)
            logger.info(f"Plotted {len(iterations)} iterations with counts: {counts}")
        else:
            logger.warning("No feedback_by_iteration data available")
            ax1.text(
                0.5,
                0.5,
                "No iteration data",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )

        # Right plot: Resolution metrics
        total_feedback = metrics.get("total_feedback_items", 0)
        addressed = metrics.get("addressed_feedback", 0)
        pending = metrics.get("pending_feedback", 0)

        logger.info(
            f"Total feedback: {total_feedback}, Addressed: {addressed}, Pending: {pending}"
        )

        if total_feedback > 0:
            data = {"Addressed": addressed, "Pending": pending}
            colors = ["#74add1", "#f46d43"]
            ax2.pie(
                data.values(),
                labels=data.keys(),
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                textprops={"fontsize": 12},
            )
            ax2.set_title(
                f"Feedback Resolution\n(Total: {total_feedback} items)",
                fontsize=15,
                fontweight="bold",
            )
        else:
            logger.warning("No feedback data available for pie chart")
            ax2.text(
                0.5,
                0.5,
                "No feedback data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved feedback resolution plot to {save_path}")
        return save_path

    def plot_convergence_metrics(
        self, metrics: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot convergence and efficiency metrics.

        Args:
            metrics: Collaboration metrics dictionary
            save_path: Optional custom save path

        Returns:
            Path to saved plot
        """
        if not save_path:
            save_path = self.output_dir / "convergence_metrics.png"

        convergence = metrics.get("convergence", {})
        time_efficiency = metrics.get("time_efficiency", {})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Convergence score
        conv_score = convergence.get("convergence_score", 0)
        feedback_addressed_pct = convergence.get("feedback_addressed_percentage", 0)

        metrics_data = {
            "Convergence\nScore": conv_score * 100,
            "Feedback\nAddressed %": feedback_addressed_pct * 100,
        }

        bars = ax1.bar(
            range(len(metrics_data)),
            list(metrics_data.values()),
            color=["#3498db", "#2ecc71"],
            alpha=0.7,
        )
        ax1.set_ylabel("Percentage (%)")
        ax1.set_title("Convergence Metrics")
        ax1.set_xticks(range(len(metrics_data)))
        ax1.set_xticklabels(list(metrics_data.keys()))
        ax1.set_ylim(0, 105)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        # Right plot: Time efficiency
        writer_time_pct = time_efficiency.get("writer_time_percentage", 0) * 100
        reviewer_time_pct = time_efficiency.get("reviewer_time_percentage", 0) * 100
        overhead_pct = 100 - writer_time_pct - reviewer_time_pct

        time_data = {
            "Writer": writer_time_pct,
            "Reviewer": reviewer_time_pct,
            "Overhead": overhead_pct,
        }

        colors = ["#3498db", "#e74c3c", "#95a5a6"]
        ax2.pie(
            time_data.values(),
            labels=time_data.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax2.set_title("Time Distribution")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved convergence metrics plot to {save_path}")
        return save_path

    def plot_feedback_type_distribution(
        self, metadata: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot distribution of feedback types across all iterations.

        Args:
            metadata: Article metadata containing feedback items
            save_path: Optional custom save path

        Returns:
            Path to saved plot
        """
        if not save_path:
            save_path = self.output_dir / "feedback_type_distribution.png"

        # Collect feedback types from all iterations (if available in metadata)
        # This would require extracting from stored feedback items
        # For now, create a placeholder structure

        logger.info(f"Saved feedback type distribution plot to {save_path}")
        return save_path

    def plot_iteration_progression(
        self, metrics: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot how metrics improve over iterations.

        Args:
            metrics: Collaboration metrics dictionary
            save_path: Optional custom save path

        Returns:
            Path to saved plot
        """
        if not save_path:
            save_path = self.output_dir / "iteration_progression.png"

        feedback_by_iter = metrics.get("feedback_by_iteration", {})
        logger.info(
            f"Plotting iteration progression. Feedback by iteration: {feedback_by_iter}"
        )

        if not feedback_by_iter:
            logger.warning(
                "No iteration data to visualize - creating empty plot with message"
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No iteration data available\n(single iteration run)",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Feedback Items")
            ax.set_title("Feedback Volume Over Iterations")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            return save_path

        iterations = sorted([int(k) for k in feedback_by_iter.keys()])
        feedback_counts = [feedback_by_iter[str(i)] for i in iterations]

        logger.info(
            f"Plotting {len(iterations)} iterations: {iterations} with counts: {feedback_counts}"
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            iterations,
            feedback_counts,
            marker="o",
            linewidth=2,
            markersize=8,
            color="#3498db",
        )
        ax.fill_between(iterations, feedback_counts, alpha=0.3, color="#3498db")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Feedback Items")
        ax.set_title("Feedback Volume Over Iterations")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, count in zip(iterations, feedback_counts):
            ax.text(i, count, str(count), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved iteration progression plot to {save_path}")
        return save_path


class ToMMetricsVisualizer:
    """Create visualizations for Theory of Mind metrics."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir) if output_dir else Path("results/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_tom_predictions(
        self, tom_metrics: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot Theory of Mind prediction metrics.

        Args:
            tom_metrics: ToM metrics dictionary
            save_path: Optional custom save path

        Returns:
            Path to saved plot
        """
        if not save_path:
            save_path = self.output_dir / "tom_predictions.png"

        total_predictions = tom_metrics.get("total_predictions", 0)
        accurate_predictions = tom_metrics.get("accurate_predictions", 0)
        accuracy_rate = tom_metrics.get("accuracy_rate", 0)

        if total_predictions == 0:
            logger.warning("No ToM predictions to visualize")
            return save_path

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Prediction accuracy
        accurate = accurate_predictions
        inaccurate = total_predictions - accurate_predictions

        data = {"Accurate": accurate, "Inaccurate": inaccurate}
        colors = ["#2ecc71", "#e74c3c"]
        ax1.pie(
            data.values(),
            labels=data.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax1.set_title(
            f"ToM Prediction Accuracy\n(Total: {total_predictions} predictions)"
        )

        # Right plot: Accuracy rate bar
        ax2.bar(["Accuracy Rate"], [accuracy_rate * 100], color="#3498db", alpha=0.7)
        ax2.set_ylabel("Percentage (%)")
        ax2.set_title("Overall ToM Prediction Accuracy")
        ax2.set_ylim(0, 105)
        ax2.grid(axis="y", alpha=0.3)

        # Add value label
        ax2.text(
            0,
            accuracy_rate * 100,
            f"{accuracy_rate*100:.1f}%",
            ha="center",
            va="bottom",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved ToM predictions plot to {save_path}")
        return save_path

    def plot_tom_interactions(
        self, tom_metrics: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot Theory of Mind interaction metrics.

        Args:
            tom_metrics: ToM metrics dictionary
            save_path: Optional custom save path

        Returns:
            Path to saved plot
        """
        if not save_path:
            save_path = self.output_dir / "tom_interactions.png"

        interactions = tom_metrics.get("interactions", 0)
        learning_occurred = tom_metrics.get("learning_occurred", 0)

        if interactions == 0:
            logger.warning("No ToM interactions to visualize")
            return save_path

        fig, ax = plt.subplots(figsize=(8, 6))

        data = {
            "Total Interactions": interactions,
            "Learning Occurred": learning_occurred,
        }

        bars = ax.bar(
            range(len(data)),
            list(data.values()),
            color=["#3498db", "#2ecc71"],
            alpha=0.7,
        )
        ax.set_ylabel("Count")
        ax.set_title("Theory of Mind Interactions")
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(list(data.keys()))
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved ToM interactions plot to {save_path}")
        return save_path


class ResearchMetricsVisualizer:
    """Create visualizations for research and retrieval metrics."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir) if output_dir else Path("results/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_research_utilization(
        self, metadata: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Path:
        """
        Plot research chunk utilization metrics.

        Args:
            metadata: Article metadata containing research metrics
            save_path: Optional custom save path

        Returns:
            Path to saved plot
        """
        if not save_path:
            save_path = self.output_dir / "research_utilization.png"

        research_metrics = metadata.get("research_metrics", {})
        if not research_metrics:
            logger.warning("No research metrics to visualize")
            return save_path

        total_chunks = research_metrics.get("total_chunks", 0)
        unique_sources = research_metrics.get("unique_sources", 0)

        if total_chunks == 0:
            return save_path

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Chunk statistics
        metrics_data = {
            "Total\nChunks": total_chunks,
            "Unique\nSources": unique_sources,
        }

        bars = ax1.bar(
            range(len(metrics_data)),
            list(metrics_data.values()),
            color=["#3498db", "#2ecc71"],
            alpha=0.7,
        )
        ax1.set_ylabel("Count")
        ax1.set_title("Research Collection Metrics")
        ax1.set_xticks(range(len(metrics_data)))
        ax1.set_xticklabels(list(metrics_data.keys()))
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        # Right plot: Source diversity (if available)
        search_summaries = research_metrics.get("search_summaries", 0)
        if search_summaries > 0:
            data = {
                "Search\nQueries": search_summaries,
                "Avg Chunks\nper Query": (
                    total_chunks / search_summaries if search_summaries > 0 else 0
                ),
            }

            bars2 = ax2.bar(
                range(len(data)),
                list(data.values()),
                color=["#e74c3c", "#f39c12"],
                alpha=0.7,
            )
            ax2.set_ylabel("Count")
            ax2.set_title("Research Efficiency")
            ax2.set_xticks(range(len(data)))
            ax2.set_xticklabels(list(data.keys()))
            ax2.grid(axis="y", alpha=0.3)

            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved research utilization plot to {save_path}")
        return save_path


def generate_collaboration_visualizations(
    results_file: Path, output_dir: Optional[Path] = None
) -> List[Path]:
    """
    Generate all collaboration metric visualizations from results file.

    Args:
        results_file: Path to results.json file
        output_dir: Optional custom output directory

    Returns:
        List of paths to generated plots
    """
    with open(results_file, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    generated_plots = []

    visualizer = CollaborationMetricsVisualizer(output_dir)

    # Process each topic's metrics
    for topic, methods in results.items():
        for method, method_data in methods.items():
            if not method_data.get("success", False):
                continue

            metadata = method_data.get("metadata", {})
            collab_metrics = metadata.get("collaboration_metrics")

            if collab_metrics:
                # Create topic-specific output directory
                topic_dir = (
                    visualizer.output_dir / f"{method}_{topic.replace(' ', '_')}"
                )
                topic_dir.mkdir(parents=True, exist_ok=True)

                # Generate plots
                try:
                    plot1 = visualizer.plot_feedback_resolution(
                        collab_metrics, topic_dir / "feedback_resolution.png"
                    )
                    plot2 = visualizer.plot_convergence_metrics(
                        collab_metrics, topic_dir / "convergence_metrics.png"
                    )
                    plot3 = visualizer.plot_iteration_progression(
                        collab_metrics, topic_dir / "iteration_progression.png"
                    )
                    generated_plots.extend([plot1, plot2, plot3])
                except Exception as e:
                    logger.error(
                        f"Failed to generate collaboration plots for {topic}: {e}"
                    )

            # Research metrics
            research_metrics = metadata.get("research_metrics")
            if research_metrics:
                topic_dir = (
                    visualizer.output_dir / f"{method}_{topic.replace(' ', '_')}"
                )
                topic_dir.mkdir(parents=True, exist_ok=True)

                research_visualizer = ResearchMetricsVisualizer(output_dir)
                try:
                    plot_research = research_visualizer.plot_research_utilization(
                        metadata, topic_dir / "research_utilization.png"
                    )
                    generated_plots.append(plot_research)
                except Exception as e:
                    logger.error(f"Failed to generate research plots for {topic}: {e}")

            # ToM metrics
            tom_metrics = metadata.get("theory_of_mind")
            if tom_metrics:
                topic_dir = (
                    visualizer.output_dir / f"{method}_{topic.replace(' ', '_')}"
                )
                topic_dir.mkdir(parents=True, exist_ok=True)

                tom_visualizer = ToMMetricsVisualizer(output_dir)
                try:
                    plot3 = tom_visualizer.plot_tom_predictions(
                        tom_metrics, topic_dir / "tom_predictions.png"
                    )
                    plot4 = tom_visualizer.plot_tom_interactions(
                        tom_metrics, topic_dir / "tom_interactions.png"
                    )
                    generated_plots.extend([plot3, plot4])
                except Exception as e:
                    logger.error(f"Failed to generate ToM plots for {topic}: {e}")

    logger.info(f"Generated {len(generated_plots)} visualization plots")
    return generated_plots


def generate_visualizations_from_experiment_dir(experiment_dir: Path) -> List[Path]:
    """
    Generate visualizations directly from an experiment directory.

    Searches for metadata files in the articles subdirectory and creates
    plots in a 'plots' subdirectory within the experiment directory.

    Args:
        experiment_dir: Path to experiment directory (e.g., results/ollama/writer_reviewer_tom_N=1_T=12.11_17:15/)

    Returns:
        List of paths to generated plots
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")

    articles_dir = experiment_dir / "articles"
    if not articles_dir.exists():
        logger.warning(f"No articles directory found in {experiment_dir}")
        return []

    # Create plots subdirectory in the experiment directory
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    generated_plots = []

    # Find all metadata files
    metadata_files = list(articles_dir.glob("*_metadata.json"))

    if not metadata_files:
        logger.warning(f"No metadata files found in {articles_dir}")
        return []

    logger.info(f"Found {len(metadata_files)} metadata file(s) in {articles_dir}")

    for metadata_file in metadata_files:
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Get metadata fields
            collab_metrics = metadata.get("collaboration_metrics")
            tom_metrics = metadata.get("theory_of_mind")
            research_metrics = metadata.get("research_metrics")

            # Generate collaboration visualizations
            if collab_metrics:
                collab_viz = CollaborationMetricsVisualizer(plots_dir)
                try:
                    logger.info(
                        f"Generating collaboration plots from {metadata_file.name}"
                    )
                    plot1 = collab_viz.plot_feedback_resolution(
                        collab_metrics, plots_dir / "feedback_resolution.png"
                    )
                    plot2 = collab_viz.plot_convergence_metrics(
                        collab_metrics, plots_dir / "convergence_metrics.png"
                    )
                    plot3 = collab_viz.plot_iteration_progression(
                        collab_metrics, plots_dir / "iteration_progression.png"
                    )
                    generated_plots.extend([plot1, plot2, plot3])
                    logger.info(
                        f"Generated {len([plot1, plot2, plot3])} collaboration plots"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate collaboration plots: {e}", exc_info=True
                    )

            # Generate ToM visualizations
            if tom_metrics and tom_metrics.get("total_predictions", 0) > 0:
                tom_viz = ToMMetricsVisualizer(plots_dir)
                try:
                    logger.info(f"Generating ToM plots from {metadata_file.name}")
                    plot4 = tom_viz.plot_tom_predictions(
                        tom_metrics, plots_dir / "tom_predictions.png"
                    )
                    plot5 = tom_viz.plot_tom_interactions(
                        tom_metrics, plots_dir / "tom_interactions.png"
                    )
                    generated_plots.extend([plot4, plot5])
                    logger.info(f"Generated {len([plot4, plot5])} ToM plots")
                except Exception as e:
                    logger.error(f"Failed to generate ToM plots: {e}", exc_info=True)

            # Generate research visualizations
            if research_metrics:
                research_viz = ResearchMetricsVisualizer(plots_dir)
                try:
                    logger.info(f"Generating research plots from {metadata_file.name}")
                    plot6 = research_viz.plot_research_utilization(
                        metadata, plots_dir / "research_utilization.png"
                    )
                    generated_plots.append(plot6)
                    logger.info("Generated research utilization plot")
                except Exception as e:
                    logger.error(
                        f"Failed to generate research plots: {e}", exc_info=True
                    )

        except Exception as e:
            logger.error(
                f"Failed to process metadata file {metadata_file}: {e}", exc_info=True
            )

    logger.info(f"Generated {len(generated_plots)} total plots in {plots_dir}")
    return generated_plots


def compare_experiments(
    experiment_dirs: List[Path], output_dir: Optional[Path] = None
) -> List[Path]:
    """
    Compare collaboration metrics across multiple experiments.

    Args:
        experiment_dirs: List of experiment directory paths
        output_dir: Optional custom output directory (defaults to first experiment's parent /comparison/)

    Returns:
        List of paths to generated comparison plots
    """
    if not output_dir:
        output_dir = experiment_dirs[0].parent / "comparison"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect metrics from all experiments
    experiment_metrics = []
    experiment_names = []

    for exp_dir in experiment_dirs:
        exp_dir = Path(exp_dir)
        articles_dir = exp_dir / "articles"

        if not articles_dir.exists():
            logger.warning(f"Skipping {exp_dir} - no articles directory")
            continue

        metadata_files = list(articles_dir.glob("*_metadata.json"))
        if not metadata_files:
            logger.warning(f"Skipping {exp_dir} - no metadata files")
            continue

        # Use first metadata file (assuming single article per experiment)
        with open(metadata_files[0], "r") as f:
            metadata = json.load(f)

        collab_metrics = metadata.get("collaboration_metrics", {})
        if collab_metrics:
            experiment_metrics.append(collab_metrics)
            experiment_names.append(exp_dir.name)

    if not experiment_metrics:
        logger.error("No valid metrics found across experiments")
        return []

    logger.info(f"Comparing {len(experiment_metrics)} experiments: {experiment_names}")

    generated_plots = []

    # Comparison Plot 1: Total iterations
    fig, ax = plt.subplots(figsize=(12, 6))
    iterations = [m.get("iterations", 0) for m in experiment_metrics]

    bars = ax.bar(
        range(len(experiment_names)), iterations, alpha=0.7, color="steelblue"
    )
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Total Iterations")
    ax.set_title("Total Iterations Across Experiments")
    ax.set_xticks(range(len(experiment_names)))
    ax.set_xticklabels(experiment_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plot1 = output_dir / "iterations_comparison.png"
    plt.savefig(plot1, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(plot1)
    logger.info(f"Saved iterations comparison to {plot1}")

    # Comparison Plot 2: Convergence score
    fig, ax = plt.subplots(figsize=(12, 6))
    convergence_scores = [
        m.get("convergence", {}).get("convergence_score", 0) * 100
        for m in experiment_metrics
    ]

    bars = ax.bar(
        range(len(experiment_names)), convergence_scores, alpha=0.7, color="#2ecc71"
    )
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Convergence Score (%)")
    ax.set_title("Convergence Score Across Experiments")
    ax.set_xticks(range(len(experiment_names)))
    ax.set_xticklabels(experiment_names, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plot2 = output_dir / "convergence_comparison.png"
    plt.savefig(plot2, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(plot2)
    logger.info(f"Saved convergence comparison to {plot2}")

    # Comparison Plot 3: Total feedback items
    fig, ax = plt.subplots(figsize=(12, 6))
    total_feedback = [m.get("total_feedback_items", 0) for m in experiment_metrics]

    bars = ax.bar(
        range(len(experiment_names)), total_feedback, alpha=0.7, color="#e74c3c"
    )
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Total Feedback Items")
    ax.set_title("Total Feedback Items Across Experiments")
    ax.set_xticks(range(len(experiment_names)))
    ax.set_xticklabels(experiment_names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plot3 = output_dir / "feedback_comparison.png"
    plt.savefig(plot3, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(plot3)
    logger.info(f"Saved feedback comparison to {plot3}")

    # Comparison Plot 4: Feedback resolution rate
    fig, ax = plt.subplots(figsize=(12, 6))
    resolution_rates = [
        m.get("convergence", {}).get("feedback_addressed_percentage", 0) * 100
        for m in experiment_metrics
    ]

    bars = ax.bar(
        range(len(experiment_names)), resolution_rates, alpha=0.7, color="#f39c12"
    )
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Feedback Resolution Rate (%)")
    ax.set_title("Feedback Resolution Rate Across Experiments")
    ax.set_xticks(range(len(experiment_names)))
    ax.set_xticklabels(experiment_names, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plot4 = output_dir / "resolution_rate_comparison.png"
    plt.savefig(plot4, dpi=150, bbox_inches="tight")
    plt.close()
    generated_plots.append(plot4)
    logger.info(f"Saved resolution rate comparison to {plot4}")

    logger.info(f"Generated {len(generated_plots)} comparison plots in {output_dir}")
    return generated_plots


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Multi-experiment comparison mode
            if len(sys.argv) < 4:
                print("Error: --compare requires at least 2 experiment directories")
                print(
                    "Usage: python collaboration_visualizations.py --compare <exp_dir1> <exp_dir2> ... [output_dir]"
                )
                sys.exit(1)

            # Find where experiment dirs end (last arg might be output_dir)
            exp_dirs = []
            output_path = None

            for arg in sys.argv[2:]:
                path = Path(arg)
                if path.is_dir() and (path / "articles").exists():
                    exp_dirs.append(path)
                elif not path.exists():
                    output_path = path
                else:
                    exp_dirs.append(path)

            if len(exp_dirs) < 2:
                print(f"Error: Found only {len(exp_dirs)} valid experiment directories")
                sys.exit(1)

            logger.info(f"Comparing {len(exp_dirs)} experiments")
            compare_experiments(exp_dirs, output_path)
        else:
            input_path = Path(sys.argv[1])

            # Check if input is a directory or results.json file
            if input_path.is_dir():
                # Treat as experiment directory
                logger.info(
                    f"Generating visualizations from experiment directory: {input_path}"
                )
                generate_visualizations_from_experiment_dir(input_path)
            elif input_path.name == "results.json" or input_path.suffix == ".json":
                # Treat as results file
                output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
                logger.info(
                    f"Generating visualizations from results file: {input_path}"
                )
                generate_collaboration_visualizations(input_path, output_path)
            else:
                print(f"Invalid input: {input_path}")
                print(
                    "Usage: python collaboration_visualizations.py <experiment_dir|results.json> [output_dir]"
                )
                print(
                    "       python collaboration_visualizations.py --compare <exp_dir1> <exp_dir2> ... [output_dir]"
                )
    else:
        print(
            "Usage: python collaboration_visualizations.py <experiment_dir|results.json> [output_dir]"
        )
        print(
            "       python collaboration_visualizations.py --compare <exp_dir1> <exp_dir2> ... [output_dir]"
        )
