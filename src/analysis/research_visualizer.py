from pathlib import Path

import logging
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

COLORBREWER_PALETTE = {
    "red1": "#a50026",
    "red2": "#d73027",
    "orange1": "#f46d43",
    "orange2": "#fdae61",
    "yellow": "#fee090",
    "cyan1": "#e0f3f8",
    "cyan2": "#abd9e9",
    "blue1": "#74add1",
    "blue2": "#4575b4",
    "blue3": "#313695",
}

METHOD_COLORS = {
    "direct": COLORBREWER_PALETTE["red1"],
    "rag": COLORBREWER_PALETTE["red2"],
    "storm": COLORBREWER_PALETTE["orange1"],
    "writer": COLORBREWER_PALETTE["blue1"],
    "writer_reviewer": COLORBREWER_PALETTE["blue2"],
    "writer_reviewer_tom": COLORBREWER_PALETTE["blue3"],
}

REFERENCE_METRICS = [
    "rouge_1",
    "rouge_l",
    "heading_soft_recall",
    "heading_entity_recall",
    "article_entity_recall",
]

JUDGE_METRICS = [
    "interest_level",
    "coherence_organization",
    "relevance_focus",
    "broad_coverage",
    "llm_judge_avg",
]


class ResearchVisualizer:
    """Publication-quality visualizations for research comparisons."""

    def __init__(self, output_base: str = "results/visualizations"):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams.update(
            {
                "font.size": 13,
                "axes.labelsize": 14,
                "axes.titlesize": 15,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.titlesize": 16,
            }
        )

    def _load_results(self, path: str) -> pd.DataFrame:
        """Load results.json and convert to DataFrame."""
        results_file = Path(path) / "results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        import json

        with open(results_file) as f:
            data = json.load(f)

        records = []

        if "results" in data:
            results_data = data["results"]
        else:
            results_data = data

        for topic, topic_methods in results_data.items():
            if topic == "summary":
                continue

            if isinstance(topic_methods, dict):
                for method, method_data in topic_methods.items():
                    if isinstance(method_data, dict):
                        record = {"method": method, "topic": topic}

                        if "evaluation" in method_data:
                            record.update(method_data["evaluation"])

                        if "llm_judge" in method_data:
                            llm_judge = method_data["llm_judge"]
                            if isinstance(llm_judge, dict):
                                for key, val in llm_judge.items():
                                    if key != "justification":
                                        record[key] = val

                                judge_scores = [
                                    llm_judge.get("interest_level", 0),
                                    llm_judge.get("coherence_organization", 0),
                                    llm_judge.get("relevance_focus", 0),
                                    llm_judge.get("broad_coverage", 0),
                                ]
                                if judge_scores:
                                    record["llm_judge_avg"] = sum(judge_scores) / len(
                                        judge_scores
                                    )

                        if "generation_time" in method_data:
                            record["generation_time"] = method_data["generation_time"]

                        if "word_count" in method_data:
                            record["word_count"] = method_data["word_count"]

                        # Extract token usage if available
                        if "token_usage" in method_data:
                            token_usage = method_data["token_usage"]
                            if isinstance(token_usage, dict):
                                record["total_tokens"] = token_usage.get(
                                    "total_tokens", 0
                                )
                                record["prompt_tokens"] = token_usage.get(
                                    "total_prompt_tokens", 0
                                )
                                record["completion_tokens"] = token_usage.get(
                                    "total_completion_tokens", 0
                                )

                        records.append(record)

        return pd.DataFrame(records)

    def _create_box_plot(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        x_var: str,
        title: str,
        output_path: Path,
        colors: Optional[Dict] = None,
        xlabel: Optional[str] = None,
        y_max: Optional[float] = None,
        y_padding: float = 0.3,
    ):
        """Create box plot showing distributions across groups."""
        n_metrics = len(metrics)
        ncols = min(3, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            # Single subplot case - axes is a single Axes object
            axes_list = [axes]  # type: ignore
        else:
            # Multiple subplots case - axes is a numpy array
            axes_list = axes.flatten()  # type: ignore

        for idx, metric in enumerate(metrics):
            ax = axes_list[idx]  # type: ignore

            metric_data = df[[x_var, metric]].dropna()
            unique_vals = sorted(metric_data[x_var].unique())

            positions = list(range(len(unique_vals)))
            box_data = [
                metric_data[metric_data[x_var] == val][metric].values
                for val in unique_vals
            ]

            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(
                    marker="D",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=6,
                ),
            )

            if colors:
                for patch, val in zip(bp["boxes"], unique_vals):
                    patch.set_facecolor(colors.get(val, COLORBREWER_PALETTE["blue1"]))
                    patch.set_alpha(0.7)
            else:
                color_cycle = [
                    COLORBREWER_PALETTE["blue3"],
                    COLORBREWER_PALETTE["blue1"],
                    COLORBREWER_PALETTE["orange1"],
                    COLORBREWER_PALETTE["red2"],
                ]
                for idx, patch in enumerate(bp["boxes"]):
                    patch.set_facecolor(color_cycle[idx % len(color_cycle)])
                    patch.set_alpha(0.7)

            # Improve label readability
            ax.set_xticks(positions)
            if len(unique_vals) > 3:
                ax.set_xticklabels(unique_vals, rotation=45, ha="right", fontsize=11)
            else:
                ax.set_xticklabels(unique_vals, rotation=0, ha="center")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xlabel(xlabel or x_var.replace("_", " ").title())
            ax.grid(axis="y", alpha=0.3)

            # Set y-axis limits with padding
            if y_max is not None:
                # User-specified max with padding
                ax.set_ylim(0, y_max)
            else:
                # Auto-calculate max from data with padding
                # Get the maximum value from all box plot elements (whiskers, outliers)
                data_max = max(max(data) if len(data) > 0 else 0 for data in box_data)

                # Add padding (default 10% of the range)
                if data_max > 0:
                    padded_max = data_max * (1 + y_padding)
                    ax.set_ylim(0, padded_max)

        for idx in range(n_metrics, len(axes_list)):
            fig.delaxes(axes_list[idx])  # type: ignore

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_path}")

    def _create_bar_plot(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        x_var: str,
        title: str,
        output_path: Path,
        colors: Optional[Dict] = None,
        xlabel: Optional[str] = None,
        y_max: Optional[float] = None,
        y_padding: float = 0.15,
    ):
        """Create bar plot showing mean values across groups."""
        n_metrics = len(metrics)
        ncols = min(3, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            # Single subplot case - axes is a single Axes object
            axes_list = [axes]  # type: ignore
        else:
            # Multiple subplots case - axes is a numpy array
            axes_list = axes.flatten()  # type: ignore

        for idx, metric in enumerate(metrics):
            ax = axes_list[idx]  # type: ignore

            metric_data = df[[x_var, metric]].dropna()
            grouped = metric_data.groupby(x_var)[metric].mean()
            unique_vals = sorted(grouped.index.tolist())

            positions = list(range(len(unique_vals)))
            means = [grouped.loc[val] for val in unique_vals]

            bar_colors = []
            for val in unique_vals:
                if colors:
                    bar_colors.append(colors.get(val, COLORBREWER_PALETTE["blue1"]))
                else:
                    color_cycle = [
                        COLORBREWER_PALETTE["blue3"],
                        COLORBREWER_PALETTE["blue1"],
                        COLORBREWER_PALETTE["orange1"],
                        COLORBREWER_PALETTE["red2"],
                    ]
                    bar_colors.append(color_cycle[len(bar_colors) % len(color_cycle)])

            bars = ax.bar(
                positions,
                means,
                alpha=0.8,
                color=bar_colors,
                edgecolor="black",
                linewidth=1.5,
            )

            # Calculate the appropriate y-axis maximum
            data_max = max(means) if means else 0

            if y_max is not None:
                plot_max = y_max
            else:
                plot_max = data_max

            # Calculate padding that accounts for text labels
            # Use minimum absolute padding to ensure text fits
            text_space = plot_max * 0.08  # Reserve 8% for text labels
            data_padding = plot_max * y_padding  # User-specified padding
            total_padding = text_space + data_padding

            # Set y-axis limits BEFORE adding text
            ax.set_ylim(0, plot_max + total_padding)

            # Add value labels on top of bars
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{mean_val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                )

            ax.set_xticks(positions)
            if len(unique_vals) > 3:
                ax.set_xticklabels(unique_vals, rotation=45, ha="right", fontsize=11)
            else:
                ax.set_xticklabels(unique_vals, rotation=0, ha="center")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_xlabel(xlabel or x_var.replace("_", " ").title())
            ax.grid(axis="y", alpha=0.3)

        for idx in range(n_metrics, len(axes_list)):
            fig.delaxes(axes_list[idx])  # type: ignore

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_path}")

    def compare_rm(self, wiki_path: str, faiss_path: str):
        """Compare wiki vs faiss retrieval methods."""
        output_dir = self.output_base / "rm_comparison"
        output_dir.mkdir(exist_ok=True)

        df_wiki = self._load_results(wiki_path)
        df_wiki["rm_type"] = "wiki"
        df_faiss = self._load_results(faiss_path)
        df_faiss["rm_type"] = "faiss"

        df = pd.concat([df_wiki, df_faiss], ignore_index=True)

        colors = {
            "wiki": COLORBREWER_PALETTE["blue2"],
            "faiss": COLORBREWER_PALETTE["orange1"],
        }

        self._create_box_plot(
            df,
            REFERENCE_METRICS,
            "rm_type",
            "Reference Metrics: Wiki vs FAISS (Box Plot)",
            output_dir / "rm_reference_metrics_box.png",
            colors,
            "Retrieval Method",
        )

        self._create_bar_plot(
            df,
            REFERENCE_METRICS,
            "rm_type",
            "Reference Metrics: Wiki vs FAISS (Bar Plot)",
            output_dir / "rm_reference_metrics_bar.png",
            colors,
            "Retrieval Method",
        )

        self._create_box_plot(
            df,
            JUDGE_METRICS,
            "rm_type",
            "LLM Judge Scores: Wiki vs FAISS (Box Plot)",
            output_dir / "rm_judge_scores_box.png",
            colors,
            "Retrieval Method",
            y_max=4,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "rm_type",
            "LLM Judge Scores: Wiki vs FAISS (Bar Plot)",
            output_dir / "rm_judge_scores_bar.png",
            colors,
            "Retrieval Method",
            y_max=4,
        )

        logger.info(f"RM comparison saved to: {output_dir}")

    def compare_writing_mode(self, section_path: str, full_path: str):
        """Compare section-by-section vs full article writing modes."""
        output_dir = self.output_base / "writing_mode_comparison"
        output_dir.mkdir(exist_ok=True)

        df_section = self._load_results(section_path)
        df_section["writing_mode"] = "section"
        df_full = self._load_results(full_path)
        df_full["writing_mode"] = "full"

        df = pd.concat([df_section, df_full], ignore_index=True)

        colors = {
            "section": COLORBREWER_PALETTE["blue2"],
            "full": COLORBREWER_PALETTE["orange1"],
        }

        self._create_box_plot(
            df,
            REFERENCE_METRICS,
            "writing_mode",
            "Reference Metrics: Section vs Full Article (Box Plot)",
            output_dir / "wm_reference_metrics_box.png",
            colors,
            "Writing Mode",
        )

        self._create_bar_plot(
            df,
            REFERENCE_METRICS,
            "writing_mode",
            "Reference Metrics: Section vs Full Article (Bar Plot)",
            output_dir / "wm_reference_metrics_bar.png",
            colors,
            "Writing Mode",
        )

        self._create_box_plot(
            df,
            JUDGE_METRICS,
            "writing_mode",
            "LLM Judge Scores: Section vs Full Article (Box Plot)",
            output_dir / "wm_judge_scores_box.png",
            colors,
            "Writing Mode",
            y_max=4,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "writing_mode",
            "LLM Judge Scores: Section vs Full Article (Bar Plot)",
            output_dir / "wm_judge_scores_bar.png",
            colors,
            "Writing Mode",
            y_max=4,
        )

        logger.info(f"Writing mode comparison saved to: {output_dir}")

    def compare_revision_mode(self, pending_path: str, section_path: str):
        """Compare pending vs section revision modes."""
        output_dir = self.output_base / "revision_mode_comparison"
        output_dir.mkdir(exist_ok=True)

        df_pending = self._load_results(pending_path)
        df_pending["revision_mode"] = "pending"
        df_section = self._load_results(section_path)
        df_section["revision_mode"] = "section"

        df = pd.concat([df_pending, df_section], ignore_index=True)

        colors = {
            "pending": COLORBREWER_PALETTE["blue2"],
            "section": COLORBREWER_PALETTE["orange1"],
        }

        self._create_box_plot(
            df,
            REFERENCE_METRICS,
            "revision_mode",
            "Reference Metrics: Pending vs Section Revision (Box Plot)",
            output_dir / "revm_reference_metrics_box.png",
            colors,
            "Revision Mode",
        )

        self._create_bar_plot(
            df,
            REFERENCE_METRICS,
            "revision_mode",
            "Reference Metrics: Pending vs Section Revision (Bar Plot)",
            output_dir / "revm_reference_metrics_bar.png",
            colors,
            "Revision Mode",
        )

        self._create_box_plot(
            df,
            JUDGE_METRICS,
            "revision_mode",
            "LLM Judge Scores: Pending vs Section Revision (Box Plot)",
            output_dir / "revm_judge_scores_box.png",
            colors,
            "Revision Mode",
            y_max=4,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "revision_mode",
            "LLM Judge Scores: Pending vs Section Revision (Bar Plot)",
            output_dir / "revm_judge_scores_bar.png",
            colors,
            "Revision Mode",
        )

        logger.info(f"Revision mode comparison saved to: {output_dir}")

    def compare_model_sweep(self, experiment_paths: List[str], sweep_type: str):
        """Compare across model sizes (writer or reviewer sweep)."""
        output_dir = self.output_base / sweep_type
        output_dir.mkdir(exist_ok=True)

        dfs = []
        for path in experiment_paths:
            df = self._load_results(path)
            # Extract model name and clean up prefixes
            model_name = Path(path).name.replace(f"{sweep_type}_", "")
            # Remove numeric prefixes like "2_" from model names
            if model_name and model_name[0].isdigit() and "_" in model_name:
                parts = model_name.split("_", 1)
                if len(parts) > 1:
                    model_name = parts[1]
            df["model"] = model_name
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # Sort models for consistent ordering
        model_size_order = sorted(df["model"].unique())
        color_palette = [
            COLORBREWER_PALETTE["blue3"],
            COLORBREWER_PALETTE["blue2"],
            COLORBREWER_PALETTE["blue1"],
            COLORBREWER_PALETTE["cyan2"],
            COLORBREWER_PALETTE["orange2"],
            COLORBREWER_PALETTE["orange1"],
            COLORBREWER_PALETTE["red2"],
            COLORBREWER_PALETTE["red1"],
        ]
        colors = dict(zip(model_size_order, color_palette[: len(model_size_order)]))

        self._create_box_plot(
            df,
            REFERENCE_METRICS,
            "model",
            f"Reference Metrics: {sweep_type.replace('_', ' ').title()} (Box Plot)",
            output_dir / f"{sweep_type}_reference_metrics_box.png",
            colors,
            "Model",
            y_max=0.7,
        )

        self._create_bar_plot(
            df,
            REFERENCE_METRICS,
            "model",
            f"Reference Metrics: {sweep_type.replace('_', ' ').title()} (Bar Plot)",
            output_dir / f"{sweep_type}_reference_metrics_bar.png",
            colors,
            "Model",
        )

        self._create_box_plot(
            df,
            JUDGE_METRICS,
            "model",
            f"LLM Judge Scores: {sweep_type.replace('_', ' ').title()} (Box Plot)",
            output_dir / f"{sweep_type}_judge_scores_box.png",
            colors,
            "Model",
            y_max=4,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "model",
            f"LLM Judge Scores: {sweep_type.replace('_', ' ').title()} (Bar Plot)",
            output_dir / f"{sweep_type}_judge_scores_bar.png",
            colors,
            "Model",
            y_max=4,
        )

        logger.info(f"{sweep_type} saved to: {output_dir}")

    def _export_method_summary_csv(self, df: pd.DataFrame, output_path: Path):
        """Export aggregated summary statistics to CSV."""
        metrics_to_aggregate = (
            REFERENCE_METRICS
            + JUDGE_METRICS
            + ["generation_time", "total_tokens", "word_count"]
        )

        # Filter to metrics that exist in the dataframe
        available_metrics = [m for m in metrics_to_aggregate if m in df.columns]

        print("DEBUG: Unique method_types in DataFrame:", df["method_type"].unique())

        summary = (
            df.groupby("method_type")[available_metrics]
            .agg(["mean", "std", "count"])
            .round(3)
        )

        # Flatten column names
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()

        summary.to_csv(output_path, index=False)
        logger.info(f"Summary CSV exported to: {output_path}")

    def _create_time_token_plots(
        self,
        df: pd.DataFrame,
        x_var: str,
        output_dir: Path,
        colors: Optional[Dict] = None,
    ):
        """Create visualizations for generation time and token consumption."""
        # Check which metrics are available
        time_metrics = []
        if "generation_time" in df.columns:
            time_metrics.append("generation_time")

        token_metrics = []
        if "total_tokens" in df.columns:
            token_metrics.append("total_tokens")
        if "prompt_tokens" in df.columns:
            token_metrics.append("prompt_tokens")
        if "completion_tokens" in df.columns:
            token_metrics.append("completion_tokens")

        # Time visualization
        if time_metrics:
            self._create_box_plot(
                df,
                time_metrics,
                x_var,
                "Generation Time by Method",
                output_dir / "methods_generation_time.png",
                colors,
                "Method",
            )

        # Token consumption visualization
        if token_metrics:
            self._create_box_plot(
                df,
                token_metrics,
                x_var,
                "Token Consumption by Method",
                output_dir / "methods_token_consumption.png",
                colors,
                "Method",
            )

        # Combined efficiency plot (time vs tokens)
        if "generation_time" in df.columns and "total_tokens" in df.columns:
            self._create_efficiency_scatter(
                df, x_var, output_dir / "methods_efficiency_scatter.png", colors
            )

    def _create_efficiency_scatter(
        self,
        df: pd.DataFrame,
        group_var: str,
        output_path: Path,
        colors: Optional[Dict] = None,
    ):
        """Create scatter plot showing efficiency (time vs tokens)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        groups = df[group_var].unique()
        for group in groups:
            group_data = df[df[group_var] == group]
            color = colors.get(group, COLORBREWER_PALETTE["blue1"]) if colors else None
            ax.scatter(
                group_data["total_tokens"],
                group_data["generation_time"],
                label=group,
                alpha=0.6,
                s=100,
                color=color,
            )

        ax.set_xlabel("Total Tokens")
        ax.set_ylabel("Generation Time (seconds)")
        ax.set_title("Generation Efficiency: Time vs Token Consumption")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Efficiency scatter plot saved to: {output_path}")

    def _create_cost_benefit_scatter(
        self,
        df: pd.DataFrame,
        group_var: str,
        output_dir: Path,
        colors: Optional[Dict] = None,
    ):
        """Create scatter plot showing cost (tokens) vs benefit (Article Entity Recall)."""

        if (
            "article_entity_recall" not in df.columns
            or "total_tokens" not in df.columns
        ):
            logger.warning("Required columns missing, skipping cost-benefit plot")
            return

        # Filter out rows with missing data
        plot_df = df.dropna(subset=["total_tokens", "article_entity_recall"]).copy()

        if len(plot_df) == 0:
            logger.warning("No valid data points found.")
            return

        # DEBUG: Print counts
        print("\n=== COST-BENEFIT SCATTER DEBUG ===")
        print("Original df method counts:")
        print(df["method_type"].value_counts())
        print("\nFiltered plot_df method counts:")
        print(plot_df["method_type"].value_counts())

        fig, ax = plt.subplots(figsize=(12, 7))

        groups = sorted(plot_df[group_var].unique())

        # Color mapping
        default_cycle = [
            COLORBREWER_PALETTE["blue3"],
            COLORBREWER_PALETTE["red2"],
            COLORBREWER_PALETTE["orange1"],
            COLORBREWER_PALETTE["blue1"],
            COLORBREWER_PALETTE["yellow"],
        ]
        color_map = {}
        for i, grp in enumerate(groups):
            if colors and grp in colors:
                color_map[grp] = colors[grp]
            else:
                color_map[grp] = default_cycle[i % len(default_cycle)]

        # Calculate means FROM PLOT_DF ONLY
        mean_data = []
        for group in groups:
            group_data = plot_df[plot_df[group_var] == group]
            if len(group_data) > 0:
                mean_tokens = group_data["total_tokens"].mean()
                mean_aer = group_data["article_entity_recall"].mean()

                # DEBUG
                print(f"\n{group}:")
                print(f"  Points: {len(group_data)}")
                print(f"  Mean tokens: {mean_tokens:.0f}")
                print(f"  Mean AER: {mean_aer:.2f}")

                mean_data.append(
                    {
                        "method": group,
                        "tokens": mean_tokens,
                        "aer": mean_aer,
                        "count": len(group_data),
                    }
                )

        # Plot individual points
        for group in groups:
            group_data = plot_df[plot_df[group_var] == group]
            grp_color = color_map[group]

            scatter = ax.scatter(
                group_data["total_tokens"],
                group_data["article_entity_recall"],
                label=f"{group} (n={len(group_data)})",  # Show count in legend
                alpha=0.6,
                s=80,
                c=[grp_color],
                edgecolors="face",
            )

        # Plot mean markers
        for item in mean_data:
            group = item["method"]
            grp_color = color_map[group]

            ax.scatter(
                [item["tokens"]],
                [item["aer"]],
                alpha=1.0,
                s=250,
                c=[grp_color],
                edgecolors="black",
                linewidths=2,
                marker="D",
                zorder=10,
            )

            ax.annotate(
                f"{item['aer']:.1f}%",
                (item["tokens"], item["aer"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
            )

        ax.set_xlabel("Total Tokens (Cost)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Article Entity Recall (%)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Cost-Benefit Analysis (n={len(plot_df)} articles)",
            fontsize=13,
            fontweight="bold",
        )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best", framealpha=0.9)

        ax.grid(alpha=0.3, linestyle="--")

        plt.tight_layout()
        save_path = output_dir / "methods_cost_benefit_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Cost-benefit scatter saved to: {save_path}")

    def compare_methods(
        self, experiment_paths: List[str], method_filter: Optional[List[str]] = None
    ):
        """Compare different methods (storm, rag, direct, writer_reviewer)."""
        output_dir = self.output_base / "method_comparison"
        output_dir.mkdir(exist_ok=True)

        dfs = []
        for path in experiment_paths:
            df = self._load_results(path)
            df["experiment"] = Path(path).name
            dfs.append(df)

        if not dfs:
            logger.warning("No results found for comparison")
            return

        df = pd.concat(dfs, ignore_index=True)

        # === Robust Method Matching Strategy ===
        # To avoid greedy substring matching (e.g. 'writer' matching 'writer_reviewer'),
        # we look for the longest matching key in METHOD_COLORS.

        def match_method_type(val):
            val_lower = str(val).lower()

            # Try exact match first
            if val_lower in METHOD_COLORS:
                return val_lower

            # Then try word boundary matching (longest first)
            sorted_keys = sorted(METHOD_COLORS.keys(), key=len, reverse=True)
            for k in sorted_keys:
                # Use word boundaries to avoid partial matches
                if re.search(rf"\b{re.escape(k)}\b", val_lower):
                    return k

            # Fallback to substring matching (longest first)
            for k in sorted_keys:
                if k in val_lower:
                    return k

            return val_lower

        df["method_type"] = df["method"].apply(match_method_type)
        print("\n=== Method Mapping Debug ===")
        print(df[["method", "method_type"]].drop_duplicates().sort_values("method"))
        print("\n=== Method Type Counts ===")
        print(df["method_type"].value_counts())

        # Apply method_filter if provided (for filtering specific methods passed via CLI)
        if method_filter:
            # Assume method_filter contains strings that should match the detected method_type
            # Normalized to match keys in METHOD_COLORS if possible
            filter_set = set(f.lower() for f in method_filter)
            df = df[df["method_type"].isin(filter_set)]

            if df.empty:
                logger.warning(
                    f"No data left after filtering for methods: {method_filter}"
                )
                return

        # Create standard visualizations
        self._create_box_plot(
            df,
            REFERENCE_METRICS,
            "method_type",
            "Reference Metrics: Method Comparison (Box Plot)",
            output_dir / "methods_reference_metrics_box.png",
            METHOD_COLORS,
            "Method",
            y_max=0.7,
        )

        self._create_bar_plot(
            df,
            REFERENCE_METRICS,
            "method_type",
            "Reference Metrics: Method Comparison (Bar Plot)",
            output_dir / "methods_reference_metrics_bar.png",
            METHOD_COLORS,
            "Method",
        )

        self._create_box_plot(
            df,
            JUDGE_METRICS,
            "method_type",
            "LLM Judge Scores: Method Comparison (Box Plot)",
            output_dir / "methods_judge_scores_box.png",
            METHOD_COLORS,
            "Method",
            y_max=4,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "method_type",
            "LLM Judge Scores: Method Comparison (Bar Plot)",
            output_dir / "methods_judge_scores_bar.png",
            METHOD_COLORS,
            "Method",
            y_max=4,
        )

        # Create Cost-Benefit Scatter Plot
        self._create_cost_benefit_scatter(
            df, "method_type", output_dir, colors=METHOD_COLORS
        )

        logger.info(f"Method comparison saved to: {output_dir}")
