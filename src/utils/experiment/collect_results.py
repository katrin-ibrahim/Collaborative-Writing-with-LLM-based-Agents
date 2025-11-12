#!/usr/bin/env python3
"""
Aggregate model-sweep results into a single README table.

Assumes folders named like: writer_sweep_<modelname>[_...]
Each folder must contain a results.json with an 'evaluation_aggregate' dict.

Usage:
    python analyze_model_sweep.py ./results_root
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.evaluation.metrics import calculate_composite_score
from src.utils.experiment.analysis_utils import extract_metrics_from_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


@dataclass
class ModelRow:
    """Stores the extracted data for a single model's results."""

    model_name: str
    metrics_raw: Dict[str, float] = field(default_factory=dict)
    composite_score: Optional[float] = None


def extract_model_name(folder_name: str, pattern: str) -> str:
    """
    Extract model name token after the experiment pattern prefix.

    The pattern may contain a wildcard, for example "writer_sweep_*".
    We take the prefix before the first '*' and use that to find the model token.
    """
    # find prefix part before any wildcard
    prefix = pattern.split("*", 1)[0].rstrip("_")
    if not prefix:
        # fallback to original folder name if pattern was just '*'
        return folder_name

    # allow an optional underscore between prefix and model token
    regex = rf"^{re.escape(prefix)}_?([a-zA-Z0-9_-]+)"
    m = re.match(regex, folder_name)
    if m:
        return m.group(1)
    return folder_name


def process_folder(folder: Path, pattern: str) -> Optional[ModelRow]:
    results_file = folder / "results.json"
    if not results_file.exists():
        logging.warning("Skipping %s: results.json not found", folder.name)
        return None

    metrics: Optional[Dict[str, float]] = extract_metrics_from_results(results_file)
    if not metrics:
        logging.warning("Skipping %s: No metrics found in results.json", folder.name)
        return None

    composite = calculate_composite_score(metrics)
    model_name = extract_model_name(folder.name, pattern)

    return ModelRow(
        model_name=model_name, metrics_raw=metrics, composite_score=composite
    )


def generate_single_table(rows: List[ModelRow], out_path: Path):
    # 1. Gather all unique metric names across all models
    all_metrics: Set[str] = set()
    for r in rows:
        all_metrics.update(r.metrics_raw.keys())

    # Remove any metric named 'composite_score' if present in raw metrics
    all_metrics.discard("composite_score")

    # Sort metric names for stable column order
    other_metrics = sorted(list(all_metrics))

    # 2. Define Table Headers
    headers = ["Model", "Composite Score"] + other_metrics

    # 3. Sort by composite_score desc (None last)
    rows_sorted = sorted(
        rows, key=lambda r: (r.composite_score is None, -(r.composite_score or 0.0))
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Model Sweep Summary\n\n")
        f.write(f"_Generated: {datetime.now().isoformat()}_\n\n")

        # header row
        f.write("| " + " | ".join(headers) + " |\n")

        # alignment row: Model left aligned, numeric columns right aligned
        # right alignment in markdown is '---:'
        align_cells = [":---"] + ["---:" for _ in headers[1:]]
        f.write("| " + " | ".join(align_cells) + " |\n")

        # 4. Determine column maxes for bolding
        col_max = {h: -float("inf") for h in headers[1:]}  # skip Model
        for r in rows_sorted:
            if r.composite_score is not None:
                col_max["Composite Score"] = max(
                    col_max["Composite Score"], r.composite_score
                )
            for metric_name in other_metrics:
                v = r.metrics_raw.get(metric_name)
                if v is not None:
                    col_max[metric_name] = max(col_max[metric_name], v)

        # 5. Write rows
        for r in rows_sorted:
            # Format Composite Score
            cs = f"{r.composite_score:.3f}" if r.composite_score is not None else "N/A"
            if (
                r.composite_score is not None
                and abs(
                    r.composite_score - col_max.get("Composite Score", -float("inf"))
                )
                < 1e-9
            ):
                cs = f"**{cs}**"

            cells = [r.model_name, cs]

            # Format raw metrics
            for metric_name in other_metrics:
                v = r.metrics_raw.get(metric_name)
                cell = f"{v:.3f}" if v is not None else "N/A"
                if (
                    v is not None
                    and abs(v - col_max.get(metric_name, -float("inf"))) < 1e-9
                ):
                    cell = f"**{cell}**"
                cells.append(cell)

            f.write("| " + " | ".join(cells) + " |\n")

        # 6. Recommendations
        f.write("\n\n## Recommendation\n\n")
        if rows_sorted and rows_sorted[0].composite_score is not None:
            best = rows_sorted[0]
            f.write(
                f"- Best model: **{best.model_name}** with Composite Score = **{best.composite_score:.3f}**\n"
            )
        else:
            f.write("- No valid Composite Score found across models.\n")

        f.write(
            "\n_Note:_ The Composite Score is calculated based on the logic in `calculate_composite_score()`.\n"
        )
    logging.info("Wrote summary markdown to %s", out_path.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate model-sweep results and generate a single-table README."
    )
    parser.add_argument(
        "root_directory",
        type=Path,
        help="Root results directory containing experiment folders.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Output markdown file (defaults to <root>/Sweep_README.md).",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="writer_sweep_*",
        help="Glob pattern to match experiment folders (default: writer_sweep_*).",
    )
    args = parser.parse_args()

    root = args.root_directory
    if not root.is_dir():
        logging.error("Path '%s' is not a valid directory.", root)
        sys.exit(1)

    out_file = args.out if args.out else (root / "Sweep_README.md")
    experiment_pattern = args.pattern

    logging.info("Scanning for '%s' in %s...", experiment_pattern, root)
    dirs = sorted(list(root.glob(experiment_pattern)))
    if not dirs:
        logging.error(
            "No experiment directories found matching '%s'.", experiment_pattern
        )
        sys.exit(1)

    rows: List[ModelRow] = []
    for d in dirs:
        if d.is_dir():
            mr = process_folder(d, experiment_pattern)
            if mr:
                rows.append(mr)

    if not rows:
        logging.error("No valid model rows parsed. Exiting.")
        sys.exit(1)

    generate_single_table(rows, out_file)
    print(f"Report generated: {out_file.resolve()}")


if __name__ == "__main__":
    main()
