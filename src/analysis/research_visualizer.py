from pathlib import Path

import logging
import matplotlib.pyplot as plt
import pandas as pd
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
    "direct": COLORBREWER_PALETTE["blue3"],
    "rag": COLORBREWER_PALETTE["blue1"],
    "storm": COLORBREWER_PALETTE["orange1"],
    "writer_reviewer": COLORBREWER_PALETTE["red2"],
    "writer_only": COLORBREWER_PALETTE["blue2"],
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
                                record["llm_judge_avg"] = sum(judge_scores) / len(
                                    judge_scores
                                )

                        if "generation_time" in method_data:
                            record["generation_time"] = method_data["generation_time"]

                        if "word_count" in method_data:
                            record["word_count"] = method_data["word_count"]

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

            # Set y-axis limit if specified
            if y_max is not None:
                ax.set_ylim(0, y_max)

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

            # Add value labels on top of bars
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (y_max or max(means)) * 0.02,
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

            # Set y-axis limit if specified
            if y_max is not None:
                ax.set_ylim(0, y_max)

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
            y_max=5,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "rm_type",
            "LLM Judge Scores: Wiki vs FAISS (Bar Plot)",
            output_dir / "rm_judge_scores_bar.png",
            colors,
            "Retrieval Method",
            y_max=5,
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
            y_max=0.7,
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
            y_max=5,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "writing_mode",
            "LLM Judge Scores: Section vs Full Article (Bar Plot)",
            output_dir / "wm_judge_scores_bar.png",
            colors,
            "Writing Mode",
            y_max=5,
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
            y_max=0.7,
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
            y_max=5,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "revision_mode",
            "LLM Judge Scores: Pending vs Section Revision (Bar Plot)",
            output_dir / "revm_judge_scores_bar.png",
            colors,
            "Revision Mode",
            y_max=5,
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
            y_max=0.7,
        )

        self._create_box_plot(
            df,
            JUDGE_METRICS,
            "model",
            f"LLM Judge Scores: {sweep_type.replace('_', ' ').title()} (Box Plot)",
            output_dir / f"{sweep_type}_judge_scores_box.png",
            colors,
            "Model",
            y_max=5,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "model",
            f"LLM Judge Scores: {sweep_type.replace('_', ' ').title()} (Bar Plot)",
            output_dir / f"{sweep_type}_judge_scores_bar.png",
            colors,
            "Model",
            y_max=5,
        )

        logger.info(f"{sweep_type} saved to: {output_dir}")

    def compare_methods(self, experiment_paths: List[str]):
        """Compare different methods (storm, rag, direct, writer_reviewer)."""
        output_dir = self.output_base / "method_comparison"
        output_dir.mkdir(exist_ok=True)

        dfs = []
        for path in experiment_paths:
            df = self._load_results(path)
            df["experiment"] = Path(path).name
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        for method_key in METHOD_COLORS.keys():
            df.loc[
                df["method"].str.contains(method_key, case=False, na=False),
                "method_type",
            ] = method_key

        if "method_type" not in df.columns:
            df["method_type"] = df["method"]

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
            y_max=5,
        )

        self._create_bar_plot(
            df,
            JUDGE_METRICS,
            "method_type",
            "LLM Judge Scores: Method Comparison (Bar Plot)",
            output_dir / "methods_judge_scores_bar.png",
            METHOD_COLORS,
            "Method",
            y_max=5,
        )
        logger.info(f"Method comparison saved to: {output_dir}")
