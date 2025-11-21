#!/usr/bin/env python3
"""
Main runner for evaluation module.
Evaluates articles from a results directory and adds evaluation metrics.
"""
import argparse
import sys
from pathlib import Path

import json
import logging
import re
from typing import Dict, List

from src.evaluation.evaluator import ArticleEvaluator
from src.evaluation.llm_judge_ollama import score_articles
from src.utils.data import FreshWikiLoader

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logger = logging.getLogger(__name__)


def normalize_string(s: str) -> str:
    """Normalize string for fuzzy matching (remove punctuation, lowercase)."""
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", "", s)
    return s.lower().strip()


def _aggregate_token_usage_for_method(method: str, token_data_list: list) -> dict:
    """Aggregate token usage data for a single method."""
    if not token_data_list:
        return {}

    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_calls = 0
    by_model_agg = {}
    by_task_agg = {}

    for token_usage in token_data_list:
        if not token_usage:
            continue
        total_tokens += token_usage.get("total_tokens", 0)
        total_prompt_tokens += token_usage.get("total_prompt_tokens", 0)
        total_completion_tokens += token_usage.get("total_completion_tokens", 0)
        total_calls += token_usage.get("total_calls", 0)

        by_model = token_usage.get("by_model", {})
        for model, model_stats in by_model.items():
            if model not in by_model_agg:
                by_model_agg[model] = {
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "calls": 0,
                    "tasks": set(),
                }
            by_model_agg[model]["tokens"] += model_stats.get("tokens", 0)
            by_model_agg[model]["prompt_tokens"] += model_stats.get("prompt_tokens", 0)
            by_model_agg[model]["completion_tokens"] += model_stats.get(
                "completion_tokens", 0
            )
            by_model_agg[model]["calls"] += model_stats.get("calls", 0)
            by_model_agg[model]["tasks"].update(model_stats.get("tasks", []))

        by_task = token_usage.get("by_task", {})
        for task, task_stats in by_task.items():
            if task not in by_task_agg:
                by_task_agg[task] = {
                    "tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "calls": 0,
                    "model": task_stats.get("model", "unknown"),
                }
            by_task_agg[task]["tokens"] += task_stats.get("tokens", 0)
            by_task_agg[task]["prompt_tokens"] += task_stats.get("prompt_tokens", 0)
            by_task_agg[task]["completion_tokens"] += task_stats.get(
                "completion_tokens", 0
            )
            by_task_agg[task]["calls"] += task_stats.get("calls", 0)

    for model_stats in by_model_agg.values():
        model_stats["tasks"] = list(model_stats["tasks"])

    num_articles = len(token_data_list)
    return {
        "method": method,
        "article_count": num_articles,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_calls": total_calls,
        "avg_tokens_per_article": (
            total_tokens / num_articles if num_articles > 0 else 0
        ),
        "avg_calls_per_article": total_calls / num_articles if num_articles > 0 else 0,
        "by_model": by_model_agg,
        "by_task": by_task_agg,
    }


def _generate_aggregation_summary(data):
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Use the pre-calculated aggregates if available (this is cleaner)
    aggregates = data.get("evaluation_aggregate", {})

    # Fallback: Collect raw lists if aggregates are missing
    methods_data = {}
    for topic, topic_data in data.get("results", {}).items():
        for method, method_data in topic_data.items():
            if method not in methods_data:
                methods_data[method] = []
            eval_results = method_data.get("evaluation")
            if (
                method_data.get("success") is True
                and eval_results
                and isinstance(eval_results, dict)
            ):
                methods_data[method].append(eval_results)

    # Get list of all methods (either from aggregates or raw data)
    all_methods = set(aggregates.keys()) | set(methods_data.keys())

    for method in all_methods:
        print(f"\n{method.upper()}:")

        # 1. Print Standard Metrics
        if method in methods_data and methods_data[method]:
            print(f"  📊 Evaluated articles: {len(methods_data[method])}")
            metrics = [
                "rouge_1",
                "rouge_l",
                "heading_soft_recall",
                "heading_entity_recall",
                "article_entity_recall",
            ]
            for metric in metrics:
                values = [r.get(metric, 0) for r in methods_data[method] if metric in r]
                if values:
                    avg_value = sum(values) / len(values)
                    print(f"  🔍 {metric}: {avg_value:.1f}")

        # 2. Print LLM Judge Metrics (from aggregates)
        if method in aggregates and "llm_judge" in aggregates[method]:
            llm_judge = aggregates[method]["llm_judge"]
            print(f"  🤖 LLM Judge ({llm_judge.get('num_articles', 0)} articles):")
            for k, v in llm_judge.items():
                if k != "num_articles":
                    print(f"     - {k}: {v:.2f}/5.0")

    print("\n" + "=" * 80)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate articles")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument(
        "--num_topics",
        "-n",
        type=int,
        default=None,
        help="Legacy arg, ignored in favor of scanning disk",
    )
    parser.add_argument("--llm_judge_model", default="qwen2.5:32b")
    parser.add_argument("--llm_judge_host", default="http://10.167.31.201:11434")
    parser.add_argument("--force_judge", action="store_true")
    parser.add_argument("--force_metrics", action="store_true")
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def save_results(results_dir: Path, data: Dict):
    with open(results_dir / "results.json", "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_results_with_disk_sync(results_dir: Path, freshwiki_entries: List) -> Dict:
    """
    1. Load existing results.json
    2. Scan disk for .md files
    3. Match disk files to FreshWiki entries (fuzzy)
    4. Update results.json with found files (setting success=True)
    """
    results_file = results_dir / "results.json"
    data = {"results": {}}
    if results_file.exists():
        try:
            with open(results_file, "r") as f:
                data = json.load(f)
        except Exception:
            logger.warning("Could not load results.json, starting fresh.")

    # Build FreshWiki lookup map: normalized_title -> Entry
    fw_map = {normalize_string(e.topic): e for e in freshwiki_entries}

    articles_dir = results_dir / "articles"
    known_methods = [
        "writer_reviewer_tom",
        "writer_reviewer",
        "writer_only",
        "collaborative",
        "direct",
        "storm",
        "rag",
    ]

    if articles_dir.exists():
        files = list(articles_dir.glob("*.md"))
        logger.info(f"Scanning {len(files)} files on disk...")

        found_count = 0
        for f in files:
            if f.name.endswith("_metadata.json"):
                continue

            # Extract method and topic part
            stem = f.stem
            method = None
            topic_part = None

            for km in known_methods:
                prefix = f"{km}_"
                if stem.startswith(prefix):
                    method = km
                    topic_part = stem[len(prefix) :]
                    break

            if not method and "_" in stem:
                try:
                    method, topic_part = stem.split("_", 1)
                except Exception:
                    pass

            if not method or not topic_part:
                continue

            # Fuzzy match topic_part to FreshWiki
            norm_key = normalize_string(topic_part)
            entry = fw_map.get(norm_key)

            if entry:
                real_topic = entry.topic  # Use official FreshWiki title

                if real_topic not in data["results"]:
                    data["results"][real_topic] = {}

                # If entry missing or failed, but file exists -> mark success
                current_entry = data["results"][real_topic].get(method, {})
                if not current_entry.get("success"):
                    # Check file size/validity
                    if f.stat().st_size > 50:
                        logger.info(
                            f"Found valid file for {method}/{real_topic}, registering in results."
                        )
                        data["results"][real_topic][method] = {
                            "success": True,
                            "article_path": str(f.relative_to(results_dir)),
                            "word_count": len(f.read_text(encoding="utf-8").split()),
                        }
                        found_count += 1
            else:
                pass

        if found_count > 0:
            logger.info(f"Synced {found_count} entries from disk to results.json")
            save_results(results_dir, data)

    return data


def main():
    args = parse_arguments()
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s"
    )
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Directory not found: {results_dir}")
        return 1

    # 1. LOAD ALL FRESHWIKI DATA
    logger.info("Loading FreshWiki dataset map...")
    freshwiki = FreshWikiLoader()
    all_entries = freshwiki.load_topics(num_topics=1000)
    if not all_entries:
        all_entries = freshwiki.load_topics(num_topics=200)

    if not all_entries:
        logger.error("Could not load FreshWiki dataset.")
        return 1

    # 2. LOAD & SYNC RESULTS
    data = load_results_with_disk_sync(results_dir, all_entries)
    results = data.get("results", {})

    # 3. EVALUATION LOOP
    evaluator = ArticleEvaluator()

    total_tasks = 0
    for topic, methods in results.items():
        for method, info in methods.items():
            if info.get("success"):
                total_tasks += 1

    logger.info(f"Found {total_tasks} successful runs to check for evaluation.")

    completed_count = 0

    for topic, topic_data in results.items():
        # Find entry
        entry = next((e for e in all_entries if e.topic == topic), None)
        if not entry:
            logger.warning(f"FreshWiki entry missing for {topic}, cannot evaluate.")
            continue

        for method, method_data in topic_data.items():
            if not isinstance(method_data, dict):
                continue

            # 1. SKIP FAILED
            if method_data.get("success") is False:
                continue

            # 2. SKIP ALREADY EVALUATED
            has_eval = "evaluation" in method_data or "metrics" in method_data
            has_judge = "llm_judge" in method_data

            skip_metrics = has_eval and not args.force_metrics
            skip_judge = has_judge and not args.force_judge

            if skip_metrics and skip_judge:
                continue

            # 3. LOAD CONTENT
            article_content = None
            if "article_path" in method_data:
                p = results_dir / method_data["article_path"]
                if p.exists():
                    article_content = p.read_text(encoding="utf-8").strip()

            if not article_content:
                logger.error(
                    f"Content missing for {method}/{topic} despite success=True. Marking failed."
                )
                method_data["success"] = False
                method_data["error"] = "File missing"
                continue

            # 4. EVALUATE
            logger.info(f"Evaluating {method}/{topic}...")
            from src.utils.data import Article

            article = Article(title=topic, content=article_content)

            if not skip_metrics:
                try:
                    res = evaluator.evaluate_article(article, entry)
                    if res:
                        method_data["evaluation"] = res
                except Exception as e:
                    logger.error(f"Metrics failed: {e}")

            if not skip_judge:
                try:
                    scores = score_articles(
                        [article_content],
                        model=args.llm_judge_model,
                        host=args.llm_judge_host,
                    )
                    if scores and "error" not in scores[0]:
                        method_data["llm_judge"] = scores[0]
                except Exception as e:
                    logger.error(f"Judge failed: {e}")

            completed_count += 1
            if completed_count % 5 == 0:
                save_results(results_dir, data)

    # Final save
    save_results(results_dir, data)

    # Aggregation
    methods_data = {}
    methods_token_data = {}
    methods_timing_data = {}
    methods_llm_judge_data = {}

    for topic, topic_data in data.get("results", {}).items():
        for method, method_data in topic_data.items():
            if not isinstance(method_data, dict):
                continue
            if method_data.get("success") is False:
                continue

            eval_results = method_data.get("evaluation")
            if eval_results and isinstance(eval_results, dict):
                methods_data.setdefault(method, []).append(eval_results)
                if "token_usage" in method_data:
                    methods_token_data.setdefault(method, []).append(
                        method_data["token_usage"]
                    )
                if "generation_time" in method_data:
                    methods_timing_data.setdefault(method, []).append(
                        method_data["generation_time"]
                    )
                if "llm_judge" in method_data:
                    methods_llm_judge_data.setdefault(method, []).append(
                        method_data["llm_judge"]
                    )

    metrics_list = [
        "rouge_1",
        "rouge_l",
        "heading_soft_recall",
        "heading_entity_recall",
        "article_entity_recall",
    ]
    llm_judge_metrics = [
        "interest_level",
        "coherence_organization",
        "relevance_focus",
        "broad_coverage",
    ]
    aggregates = {}

    for method, res_list in methods_data.items():
        agg = {}
        for metric in metrics_list:
            vals = [r.get(metric, 0.0) for r in res_list if metric in r]
            if vals:
                agg[metric] = sum(vals) / len(vals)

        if method in methods_llm_judge_data:
            llm_judge_list = methods_llm_judge_data[method]
            llm_judge_agg = {}
            for metric in llm_judge_metrics:
                vals = [r.get(metric, 0) for r in llm_judge_list if metric in r]
                if vals:
                    llm_judge_agg[metric] = sum(vals) / len(vals)
            if llm_judge_agg:
                llm_judge_agg["num_articles"] = len(llm_judge_list)
                agg["llm_judge"] = llm_judge_agg

        if method in methods_token_data:
            agg["token_usage"] = _aggregate_token_usage_for_method(
                method, methods_token_data[method]
            )

        if method in methods_timing_data:
            timing = methods_timing_data[method]
            agg["avg_generation_time"] = sum(timing) / len(timing)

        if agg:
            aggregates[method] = agg

    if aggregates:
        data["evaluation_aggregate"] = aggregates
        save_results(results_dir, data)

    _generate_aggregation_summary(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
