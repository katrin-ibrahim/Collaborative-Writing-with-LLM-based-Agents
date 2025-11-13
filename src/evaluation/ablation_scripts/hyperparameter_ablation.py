#!/usr/bin/env python3
"""
Automated Sequential Ablation Pipeline
Runs ablations in sequence, evaluates results, and automatically selects best config for next phase.

Usage:
    python scripts/sequential_ablation.py --num-topics 10
"""

import argparse
import subprocess
import sys
from pathlib import Path

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from evaluation.metrics import calculate_composite_score
from utils.experiment.analysis_utils import extract_metrics_from_results

# Ensure project root is on sys.path so absolute imports from the `src` package resolve when running this script


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Result from a single phase."""

    phase: int
    experiments: List[str]
    best_config: str
    best_value: float
    composite_score: float
    all_scores: Dict[str, float]


class SequentialAblationPipeline:
    """Automated sequential ablation with auto-evaluation and decision making."""

    def __init__(
        self,
        num_topics: int = 10,
        backend: str = "ollama",
        model_config: str = "balanced_writer",
        output_dir: str = "results/ollama/sequential_ablation",
    ):
        self.num_topics = num_topics
        self.backend = backend
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track best configs from each phase
        self.best_rm = None
        self.best_writing_mode = None
        self.best_revision_mode = None
        self.best_grounding = None

        self.phase_results: List[PhaseResult] = []

    def run_experiment(
        self,
        experiment_name: str,
        method: str,
        rm: Optional[str] = None,
        writing_mode: Optional[str] = None,
        revise_mode: Optional[str] = None,
        max_iterations: int = 3,
    ) -> Optional[Path]:
        """Run a single experiment and return results directory."""

        cmd = [
            "python",
            "-m",
            "src.main",
            "--methods",
            method,
            "--num_topics",
            str(self.num_topics),
            "--model_config",
            self.model_config,
            "--backend",
            self.backend,
            "--experiment_name",
            experiment_name,
            "--override_model",
            "qwen2.5:32b",
        ]

        if rm:
            cmd.extend(["--retrieval_manager", rm])
        if writing_mode:
            cmd.extend(["--writing_mode", writing_mode])
        if revise_mode:
            cmd.extend(["--revise_mode", revise_mode])
        if max_iterations:
            cmd.extend(["--max_iterations", str(max_iterations)])

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            # Stream output in real-time instead of capturing it
            result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout

            if result.returncode != 0:
                logger.error(f"Experiment failed: {experiment_name}")
                return None

            # Find results directory
            results_dir = Path("results") / self.backend / experiment_name
            if not results_dir.exists():
                logger.error(f"Results directory not found: {results_dir}")
                return None

            eval_cmd = [
                "python",
                "-m",
                "src.evaluation",
                str(results_dir),
            ]

            # Stream output in real-time
            result = subprocess.run(eval_cmd, timeout=3600)

            if result.returncode != 0:
                logger.error(f"Evaluation failed for {results_dir}")

            logger.info(f"Evaluation completed for {results_dir}")

            return results_dir

        except subprocess.TimeoutExpired:
            logger.error(f"Experiment timeout: {experiment_name}")
            return None
        except Exception as e:
            logger.error(f"Experiment error: {e}")
            return None

    def rm_selection(self) -> PhaseResult:
        """
        Phase 1: Test faiss vs wiki retrieval managers.

        Note: GDELT news retrieval was tested but removed due to poor quality:
        - Irrelevant content (e.g., celebrity news for sports queries)
        - Non-English articles passing through filters
        - Low extraction success rate (~25%)
        Wiki RM won with composite score 37.51 vs FAISS 36.29.
        """

        logger.info("=" * 80)
        logger.info("PHASE 1: Retrieval Manager Selection (faiss vs wiki)")
        logger.info("=" * 80)

        experiments = {
            "faiss": "hyperparameter_rm_faiss",
            "wiki": "hyperparameter_rm_wiki",
        }

        scores = {}

        for rm, exp_name in experiments.items():
            logger.info(f"\nTesting RM: {rm}")

            # Check if experiment already exists
            existing_results = (
                Path("results") / self.backend / exp_name / "results.json"
            )
            if existing_results.exists():
                logger.info(f"  Found existing results for {rm}, loading metrics...")
                metrics = extract_metrics_from_results(existing_results)
                if metrics:
                    composite = calculate_composite_score(metrics)
                    scores[rm] = composite
                    logger.info(
                        f"  {rm}: Composite Score = {composite:.2f} (from existing results)"
                    )
                    logger.info(f"    ROUGE-1: {metrics.get('rouge_1', 0):.2f}")
                    logger.info(f"    LLM Judge: {metrics.get('llm_judge_avg', 0):.2f}")
                    continue
                else:
                    logger.warning(
                        "  Could not extract metrics from existing results, re-running..."
                    )

            # Run experiment
            results_dir = self.run_experiment(
                experiment_name=exp_name,
                method="writer_v3",
                rm=rm,
            )

            if results_dir:
                metrics = extract_metrics_from_results(results_dir / "results.json")
                if metrics:
                    composite = calculate_composite_score(metrics)
                    scores[rm] = composite
                    logger.info(f"  {rm}: Composite Score = {composite:.2f}")
                    logger.info(f"    ROUGE-1: {metrics.get('rouge_1', 0):.2f}")
                    logger.info(f"    LLM Judge: {metrics.get('llm_judge_avg', 0):.2f}")

        if not scores:
            logger.error("No valid scores from Phase 1!")
            sys.exit(1)

        # Select best
        best_rm = max(scores, key=lambda k: scores[k])
        self.best_rm = best_rm

        result = PhaseResult(
            phase=1,
            experiments=list(experiments.values()),
            best_config="retrieval_manager",
            best_value=best_rm,
            composite_score=scores[best_rm],
            all_scores=scores,
        )

        self.phase_results.append(result)
        logger.info(
            f"\n✓ Phase 1 Decision: retrieval_manager = {best_rm} (score: {scores[best_rm]:.2f})"
        )

        return result

    def writing_mode_selection(self) -> PhaseResult:
        """Phase 2: Test section vs full_article writing modes."""

        logger.info("=" * 80)
        logger.info("PHASE 2: Writing Mode Selection (section vs full_article)")
        logger.info(f"Using RM: {self.best_rm}")
        logger.info("=" * 80)

        experiments = {
            "section": "hyperparameter_wm_section",
            "full_article": "hyperparameter_wm_full_article",
        }

        scores = {}

        for wm, exp_name in experiments.items():
            logger.info(f"\nTesting writing_mode: {wm}")

            results_dir = self.run_experiment(
                experiment_name=exp_name,
                method="writer_v3",
                rm=self.best_rm,
                writing_mode=wm,
            )

            if results_dir:
                metrics = extract_metrics_from_results(results_dir / "results.json")
                if metrics:
                    composite = calculate_composite_score(metrics)
                    scores[wm] = composite
                    logger.info(f"  {wm}: Composite Score = {composite:.2f}")
                    logger.info(f"    ROUGE-1: {metrics.get('rouge_1', 0):.2f}")
                    logger.info(f"    LLM Judge: {metrics.get('llm_judge_avg', 0):.2f}")

        if not scores:
            logger.error("No valid scores from Phase 2!")
            sys.exit(1)

        best_wm = max(scores, key=lambda k: scores[k])
        self.best_writing_mode = best_wm

        result = PhaseResult(
            phase=2,
            experiments=list(experiments.values()),
            best_config="writing_mode",
            best_value=best_wm,
            composite_score=scores[best_wm],
            all_scores=scores,
        )

        self.phase_results.append(result)
        logger.info(
            f"\n✓ Phase 2 Decision: writing_mode = {best_wm} (score: {scores[best_wm]:.2f})"
        )

        return result

    def revision_mode_selection(self) -> PhaseResult:
        """Phase 3: Test single_section vs pending_sections revision modes (ungrounded reviewer)."""

        logger.info("=" * 80)
        logger.info(
            "PHASE 3: Revision Mode Selection (single_section vs pending_sections)"
        )
        logger.info(f"Using RM: {self.best_rm}, Writing Mode: {self.best_writing_mode}")
        logger.info("Reviewer: ungrounded")
        logger.info("=" * 80)

        experiments = {
            "section": "hyperparameter_revm_section",
            "pending": "hyperparameter_revm_pending",
        }

        scores = {}

        for revm, exp_name in experiments.items():
            logger.info(f"\nTesting revise_mode: {revm}")

            results_dir = self.run_experiment(
                experiment_name=exp_name,
                method="writer_reviewer",
                rm=self.best_rm,
                writing_mode=self.best_writing_mode,
                revise_mode=revm,
                max_iterations=3,
            )

            if results_dir:
                metrics = extract_metrics_from_results(results_dir / "results.json")
                if metrics:
                    composite = calculate_composite_score(metrics)
                    scores[revm] = composite
                    logger.info(f"  {revm}: Composite Score = {composite:.2f}")
                    logger.info(f"    ROUGE-1: {metrics.get('rouge_1', 0):.2f}")
                    logger.info(f"    LLM Judge: {metrics.get('llm_judge_avg', 0):.2f}")

        if not scores:
            logger.error("No valid scores from Phase 3!")
            sys.exit(1)

        best_revm = max(scores, key=lambda k: scores[k])
        self.best_revision_mode = best_revm

        result = PhaseResult(
            phase=3,
            experiments=list(experiments.values()),
            best_config="revise_mode",
            best_value=best_revm,
            composite_score=scores[best_revm],
            all_scores=scores,
        )

        self.phase_results.append(result)
        logger.info(
            f"\n✓ Phase 3 Decision: revise_mode = {best_revm} (score: {scores[best_revm]:.2f})"
        )

        return result

    def save_final_summary(self):
        """Save final summary of all phases."""

        summary = {
            "pipeline_config": {
                "num_topics": self.num_topics,
                "backend": self.backend,
                "model_config": self.model_config,
            },
            "final_best_config": {
                "retrieval_manager": self.best_rm,
                "writing_mode": self.best_writing_mode,
                "revise_mode": self.best_revision_mode,
                "reviewer_grounding": self.best_grounding,
            },
            "phase_results": [
                {
                    "phase": r.phase,
                    "best_config": r.best_config,
                    "best_value": r.best_value,
                    "composite_score": r.composite_score,
                    "all_scores": r.all_scores,
                }
                for r in self.phase_results
            ],
        }

        summary_file = self.output_dir / "hyperparameter_ablation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nFinal summary saved to: {summary_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("SEQUENTIAL ABLATION PIPELINE - FINAL SUMMARY")
        print("=" * 80)
        print("\nOptimal Configuration Found:")
        print(f"  Retrieval Manager: {self.best_rm}")
        print(f"  Writing Mode: {self.best_writing_mode}")
        print(f"  Revision Mode: {self.best_revision_mode}")
        print(f"  Reviewer Grounding: {self.best_grounding}")
        print("\nPhase-by-Phase Decisions:")
        for r in self.phase_results:
            print(f"\n  Phase {r.phase}: {r.best_config} = {r.best_value}")
            print(f"    Composite Score: {r.composite_score:.2f}")
            print(f"    All Options: {r.all_scores}")
        print("\n" + "=" * 80)

    def run_all(self):
        """Run all phases sequentially."""

        phases = [
            (1, self.rm_selection),
            (2, self.writing_mode_selection),
            (3, self.revision_mode_selection),
        ]

        for phase_num, phase_func in phases:
            try:
                phase_func()
            except Exception as e:
                logger.error(f"Phase {phase_num} failed: {e}")
                raise

        self.save_final_summary()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequential ablation pipeline with auto-evaluation and decision making"
    )

    parser.add_argument(
        "--num-topics",
        "-n",
        type=int,
        default=10,
        help="Number of topics per experiment",
    )

    parser.add_argument(
        "--backend", "-b", default="ollama", help="Backend to use (ollama/slurm)"
    )

    parser.add_argument(
        "--model-config",
        "-mc",
        default="balanced_writer",
        help="Model configuration name",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        default="results/sequential_ablation",
        help="Output directory for summary",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = SequentialAblationPipeline(
        num_topics=args.num_topics,
        backend=args.backend,
        model_config=args.model_config,
        output_dir=args.output_dir,
    )

    pipeline.run_all()

    logger.info("\n✓ Sequential ablation pipeline complete!")


if __name__ == "__main__":
    main()
