#!/usr/bin/env python3
"""
Main runner for evaluation module.
Evaluates articles from a results directory and adds evaluation metrics.

Updated to use consolidated metrics functions instead of class-based approach.
"""
import argparse
import sys
import time
from pathlib import Path

import json
import logging
from typing import Dict, Optional

from src.evaluation.evaluator import ArticleEvaluator
from src.evaluation.llm_judge_ollama import score_articles
from src.utils.data import FreshWikiLoader
from src.utils.experiment import save_final_results

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


logger = logging.getLogger(__name__)


# Removed incomplete multiprocessing implementation
# Multiprocessing would be slower due to model loading overhead in each process


def _aggregate_token_usage_for_method(method: str, token_data_list: list) -> dict:
    """Aggregate token usage data for a single method."""
    if not token_data_list:
        return {}

    # Initialize aggregation
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_calls = 0
    by_model_agg = {}
    by_task_agg = {}

    # Aggregate across all topics for this method
    for token_usage in token_data_list:
        if not token_usage:
            continue

        # Aggregate totals
        total_tokens += token_usage.get("total_tokens", 0)
        total_prompt_tokens += token_usage.get("total_prompt_tokens", 0)
        total_completion_tokens += token_usage.get("total_completion_tokens", 0)
        total_calls += token_usage.get("total_calls", 0)

        # Aggregate by model
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

        # Aggregate by task
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

    # Convert sets to lists for JSON serialization
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
    """Generate and log aggregation summary of evaluation results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    methods_data = {}

    # Collect all evaluation results by method (only successful generations)
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

    # Calculate and display averages for each method
    for method, results_list in methods_data.items():
        if not results_list:
            continue

        print(f"\n{method.upper()}:")
        print(f"  📊 Evaluated articles: {len(results_list)}")

        # Calculate averages for STORM metrics
        metrics = [
            "rouge_1",
            "rouge_l",
            "heading_soft_recall",
            "heading_entity_recall",
            "article_entity_recall",
        ]

        for metric in metrics:
            values = [r.get(metric, 0) for r in results_list if metric in r]
            if values:
                avg_value = sum(values) / len(values)
                metric_display = {
                    "rouge_1": "ROUGE-1",
                    "rouge_l": "ROUGE-L",
                    "heading_soft_recall": "HSR (Heading Soft Recall)",
                    "heading_entity_recall": "HER (Heading Entity Recall)",
                    "article_entity_recall": "AER (Article Entity Recall)",
                }
                print(f"  🔍 {metric_display[metric]}: {avg_value:.1f}")

        # Show LLM judge scores if available
        eval_aggregate = data.get("evaluation_aggregate", {})
        if method in eval_aggregate:
            method_agg = eval_aggregate[method]
            llm_judge_data = method_agg.get("llm_judge")
            if llm_judge_data:
                print(
                    f"  📝 LLM Judge Scores ({llm_judge_data.get('num_articles', 0)} articles):"
                )
                print(
                    f"     Interest Level: {llm_judge_data.get('interest_level', 0):.2f}/5"
                )
                print(
                    f"     Coherence & Organization: {llm_judge_data.get('coherence_organization', 0):.2f}/5"
                )
                print(
                    f"     Relevance & Focus: {llm_judge_data.get('relevance_focus', 0):.2f}/5"
                )
                print(
                    f"     Broad Coverage: {llm_judge_data.get('broad_coverage', 0):.2f}/5"
                )

                # Calculate overall average
                llm_avg = (
                    sum(
                        [
                            llm_judge_data.get("interest_level", 0),
                            llm_judge_data.get("coherence_organization", 0),
                            llm_judge_data.get("relevance_focus", 0),
                            llm_judge_data.get("broad_coverage", 0),
                        ]
                    )
                    / 4
                )
                print(f"     Overall Average: {llm_avg:.2f}/5 ({llm_avg * 20:.1f}%)")

        # Show token usage and timing information from aggregated data
        if method in eval_aggregate:
            method_agg = eval_aggregate[method]

            # Display timing information
            if "avg_generation_time" in method_agg:
                avg_time = method_agg["avg_generation_time"]
                total_time = method_agg.get("total_generation_time", 0)
                print(
                    f"  ⏱️ Avg generation time: {avg_time:.1f}s (total: {total_time:.1f}s)"
                )

            # Display token usage information
            token_usage = method_agg.get("token_usage", {})
            if token_usage:
                avg_tokens = token_usage.get("avg_tokens_per_article", 0)
                total_tokens = token_usage.get("total_tokens", 0)
                avg_calls = token_usage.get("avg_calls_per_article", 0)
                print(
                    f"  🔢 Avg tokens per article: {avg_tokens:.0f} (total: {total_tokens:,})"
                )
                print(f"  📞 Avg API calls per article: {avg_calls:.1f}")

                # Show breakdown by model for multi-model methods
                by_model = token_usage.get("by_model", {})
                if len(by_model) > 1:
                    print("  📋 Token usage by model:")
                    for model, stats in by_model.items():
                        tokens = stats.get("tokens", 0)
                        calls = stats.get("calls", 0)
                        tasks = stats.get("tasks", [])
                        print(
                            f"    • {model}: {tokens:,} tokens, {calls} calls ({len(tasks)} tasks)"
                        )

    print("\n" + "=" * 80)


def parse_arguments():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate articles from a results directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate articles in a specific results directory
  %(prog)s results/ollama/run_20250709_104159

  # Evaluate with debug logging
  %(prog)s results/ollama/run_20250709_104159 --log_level DEBUG
        """,
    )

    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to the results directory containing articles and results.json",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-evaluation even if evaluation results already exist",
    )
    parser.add_argument(
        "--llm_judge",
        action="store_true",
        help="[DEPRECATED] LLM judge now runs unconditionally. This flag is ignored.",
    )
    parser.add_argument(
        "--llm_judge_model",
        default="qwen2.5:32b",
        help="Ollama model for LLM judge (default: qwen2.5:32b). Recommended: qwen2.5:32b, deepseek-r1:14b, or gemma3:27b",
    )
    parser.add_argument(
        "--llm_judge_host",
        default="http://10.167.31.201:11434",
        help="Ollama server URL (default: UKP server http://10.167.31.201:11434)",
    )

    return parser.parse_args()


def load_results(results_dir: Path) -> Dict:
    """Load existing results from results.json.

    If missing, reconstruct it by scanning the articles directory and using
    save_final_results to persist a proper file (with metadata if available).
    """
    results_file = results_dir / "results.json"
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
        # If results.json exists but lacks a usable "results" block, attempt reconstruction
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("results"), dict)
            or not data.get("results")
        ):
            logger.warning(
                "Existing results.json has no results; attempting reconstruction from articles..."
            )
            # fall through to reconstruction path below
        else:
            return data

    logger.warning(f"Results file not found: {results_file}")

    # Attempt reconstruction from articles directory
    articles_dir = results_dir / "articles"
    if not articles_dir.exists():
        logger.error("No articles directory to reconstruct from. Run generation first.")
        return {}

    # Infer methods/topics using longest-known method prefix matching
    known_methods = [
        "writer_reviewer_tom",
        "writer_reviewer",
        "writer_only",
        "collaborative",
        "direct",
        "storm",
        "rag",
    ]
    methods = set()
    topics = set()
    for article_file in articles_dir.glob("*.md"):
        if article_file.name.endswith("_metadata.json"):
            continue
        stem = article_file.stem
        matched = False
        for km in known_methods:
            prefix = f"{km}_"
            if stem.startswith(prefix):
                methods.add(km)
                topics.add(stem[len(prefix) :].replace("_", " "))
                matched = True
                break
        if not matched and "_" in stem:
            # Fallback: split on first underscore
            method, rest = stem.split("_", 1)
            methods.add(method)
            topics.add(rest.replace("_", " "))

    if not methods or not topics:
        logger.error("Could not infer methods/topics from articles directory.")
        return {}

    logger.info(
        f"Reconstructing results.json from articles (methods={sorted(methods)}, topics={len(topics)})"
    )
    try:
        ok = save_final_results(
            results_dir, sorted(topics), sorted(methods), total_time=0.0
        )
        if not ok:
            logger.error("Failed to save reconstructed results.json")
            return {}
        with open(results_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        return {}


def save_results(results_dir: Path, data: Dict):
    """Save updated results back to results.json."""
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2, default=str)


def get_article_content(results_dir: Path, method: str, topic: str) -> Optional[str]:
    """Get article content from file without storing in memory."""
    articles_dir = results_dir / "articles"

    # Normalize writer-prefixed topic keys for non-writer methods
    def _strip_writer_prefixes(t: str) -> str:
        for pref in ("reviewer tom ", "reviewer ", "only "):
            if t.lower().startswith(pref):
                return t[len(pref) :]
        return t

    non_writer_methods = {"direct", "rag", "storm"}
    if method in non_writer_methods:
        topic = _strip_writer_prefixes(topic)

    # Normalize topic name (replace spaces with underscores for filenames)
    topic_normalized = topic.replace(" ", "_")
    topic_normalized = topic_normalized.replace("/", "_")  # Handle slashes too

    # Try different filename patterns
    possible_files = [
        articles_dir / f"{method}_{topic_normalized}.md",
        articles_dir / f"{method}_{topic}.md",  # Original with spaces
    ]

    for article_file in possible_files:
        if article_file.exists():
            try:
                with open(article_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    logger.debug(f"Found article content at {article_file}")
                    return content
            except Exception as e:
                logger.warning(f"Failed to read {article_file}: {e}")
                continue

    logger.warning(f"No article content found for {method}/{topic}")
    logger.info(f"Tried files: {[str(f) for f in possible_files]}")
    return None


def main():
    """Main evaluation function."""
    args = parse_arguments()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    logger.info(f"Starting evaluation of results in: {results_dir}")
    start_time = time.time()

    try:
        # Load existing results
        logger.info("Loading existing results...")
        data = load_results(results_dir)

        # Resume mode: if some evaluations already exist and --force is not set,
        # continue evaluating only the missing pairs (no early exit).
        if not args.force:
            logger.info("Resume mode: will skip already-evaluated pairs.")

        # Load FreshWiki dataset for evaluation
        logger.info("Loading FreshWiki dataset...")
        freshwiki = FreshWikiLoader()
        entries = freshwiki.load_topics(1000)  # Load all available entries

        if not entries:
            logger.error("No FreshWiki entries loaded")
            return 1

        logger.info(f"Loaded {len(entries)} FreshWiki entries")

        # Create evaluator
        evaluator = ArticleEvaluator()

        # Process each topic and method
        results = data.get("results", {})
        total_evaluations = 0
        successful_evaluations = 0
        skipped_missing = 0
        skipped_existing = 0

        # Pre-compute how many evaluations we will actually run (for ETA)
        planned_evaluations = 0
        for _topic, _topic_data in results.items():
            for _method, _method_data in _topic_data.items():
                if not isinstance(_method_data, dict):
                    continue
                if not args.force and (
                    "evaluation" in _method_data or "metrics" in _method_data
                ):
                    continue
                planned_evaluations += 1

        eval_loop_start = time.time()
        completed_evaluations = 0
        # Auto-save frequency (evaluations). Set to 1 to save after each evaluation.
        autosave_every = 1

        def _fmt_secs(secs: float) -> str:
            secs = max(0, int(secs))
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h}h{m:02d}m{s:02d}s"
            if m:
                return f"{m}m{s:02d}s"
            return f"{s}s"

        def _mark_progress():
            nonlocal completed_evaluations
            if planned_evaluations <= 0:
                return
            completed_evaluations += 1
            elapsed = time.time() - eval_loop_start
            rate = completed_evaluations / elapsed if elapsed > 0 else 0.0
            remaining = planned_evaluations - completed_evaluations
            eta = (remaining / rate) if rate > 0 else 0.0
            pct = (completed_evaluations / planned_evaluations) * 100.0
            logger.info(
                f"Progress: {completed_evaluations}/{planned_evaluations} ({pct:.1f}%) | "
                f"elapsed {_fmt_secs(elapsed)} | ETA {_fmt_secs(eta)}"
            )

        for topic, topic_data in results.items():
            logger.info(f"Processing topic: {topic}")

            # Find matching FreshWiki entry
            freshwiki_entry = None
            for entry in entries:
                # Normalize both titles by replacing underscores with spaces for comparison
                entry_topic_normalized = entry.topic.replace("_", " ").lower()
                topic_normalized = topic.replace("_", " ").lower()

                if (
                    entry_topic_normalized == topic_normalized
                    or topic_normalized in entry_topic_normalized
                    or entry_topic_normalized in topic_normalized
                ):
                    freshwiki_entry = entry
                    break

            if not freshwiki_entry:
                logger.warning(f"No FreshWiki entry found for topic: {topic}")
                continue

            for method, method_data in topic_data.items():
                if not isinstance(method_data, dict):
                    continue

                logger.debug(f"Evaluating {method} for {topic}")

                # Skip if already evaluated and not forcing
                if not args.force and (
                    "evaluation" in method_data or "metrics" in method_data
                ):
                    logger.debug(f"Skipping {method}/{topic} - already evaluated")
                    skipped_existing += 1
                    continue

                # Get article content
                article_content = None

                # Try to get content from stored path
                if "article_path" in method_data:
                    article_file = results_dir / method_data["article_path"]
                    if article_file.exists():
                        try:
                            with open(article_file, "r", encoding="utf-8") as f:
                                article_content = f.read().strip()
                        except Exception as e:
                            logger.warning(
                                f"Failed to read article from path {article_file}: {e}"
                            )

                # Fallback to search for article file
                if not article_content:
                    article_content = get_article_content(results_dir, method, topic)

                if not article_content:
                    logger.error(f"Could not find article content for {method}/{topic}")
                    # Record error only; do not create a zeroed 'evaluation' block
                    method_data["evaluation_error"] = "Could not find article content"
                    skipped_missing += 1
                    _mark_progress()
                    continue

                # Create Article object
                from src.utils.data import Article

                article = Article(title=topic, content=article_content)

                # Evaluate the article
                try:
                    logger.debug(f"Running evaluation for {method}/{topic}")
                    total_evaluations += 1
                    eval_results = evaluator.evaluate_article(article, freshwiki_entry)

                    if eval_results:
                        method_data["evaluation"] = eval_results
                        successful_evaluations += 1

                        # Ensure word count is present
                        if "word_count" not in method_data:
                            method_data["word_count"] = len(article_content.split())

                        logger.debug(f"✅ Evaluation completed for {method}/{topic}")
                        logger.debug(
                            f"   ROUGE-1: {eval_results.get('rouge_1', 0):.1f}%"
                        )
                        logger.debug(
                            f"   HSR: {eval_results.get('heading_soft_recall', 0):.1f}%"
                        )
                        logger.debug(
                            f"   HER: {eval_results.get('heading_entity_recall', 0):.1f}%"
                        )
                        logger.debug(
                            f"   AER: {eval_results.get('article_entity_recall', 0):.1f}%"
                        )

                        # Run LLM judge unconditionally
                        try:
                            logger.debug(f"Running LLM judge for {method}/{topic}")
                            llm_results = score_articles(
                                article_texts=[article_content],
                                model=args.llm_judge_model,
                                host=args.llm_judge_host,
                                temperature=0.0,
                            )
                            if llm_results and "error" not in llm_results[0]:
                                method_data["llm_judge"] = llm_results[0]
                                logger.debug(
                                    f"✅ LLM judge completed for {method}/{topic}"
                                )
                                logger.debug(
                                    f"   Interest: {llm_results[0].get('interest_level', 0)}/5"
                                )
                                logger.debug(
                                    f"   Coherence: {llm_results[0].get('coherence_organization', 0)}/5"
                                )
                            else:
                                logger.warning(
                                    f"LLM judge failed for {method}/{topic}: {llm_results[0].get('error', 'Unknown error')}"
                                )
                        except Exception as e:
                            logger.warning(f"LLM judge error for {method}/{topic}: {e}")
                    else:
                        logger.error(f"Evaluation returned None for {method}/{topic}")
                        method_data["evaluation_error"] = "Evaluation returned None"

                except Exception as e:
                    logger.error(f"Evaluation failed for {method}/{topic}: {e}")
                    # Record error only; do not create a zeroed 'evaluation' block
                    method_data["evaluation_error"] = str(e)
                finally:
                    # Count this pair as completed (success or failure)
                    _mark_progress()
                    # Periodically persist progress for resume capability
                    try:
                        if (
                            autosave_every == 1
                            or (completed_evaluations % autosave_every) == 0
                        ):
                            save_results(results_dir, data)
                    except Exception as _e:
                        logger.debug(f"Autosave failed: {_e}")

        # Update results data with evaluation timestamp
        data["evaluation_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data["evaluation_stats"] = {
            "total_evaluations": total_evaluations,
            "successful_evaluations": successful_evaluations,
            "success_rate": (
                successful_evaluations / total_evaluations
                if total_evaluations > 0
                else 0.0
            ),
            "skipped_missing": skipped_missing,
            "skipped_existing": skipped_existing,
        }

        # Compute and persist aggregation by method
        try:
            methods_data = {}
            methods_token_data = {}
            methods_timing_data = {}
            methods_llm_judge_data = {}

            for topic, topic_data in data.get("results", {}).items():
                for method, method_data in topic_data.items():
                    if not isinstance(method_data, dict):
                        continue
                    eval_results = method_data.get("evaluation")
                    if (
                        method_data.get("success") is True
                        and eval_results
                        and isinstance(eval_results, dict)
                    ):
                        methods_data.setdefault(method, []).append(eval_results)

                        # Collect token usage data if available
                        token_usage = method_data.get("token_usage")
                        if token_usage:
                            methods_token_data.setdefault(method, []).append(
                                token_usage
                            )

                        # Collect timing data
                        gen_time = method_data.get("generation_time", 0.0)
                        if gen_time:
                            methods_timing_data.setdefault(method, []).append(gen_time)

                        # Collect LLM judge data if available
                        llm_judge_result = method_data.get("llm_judge")
                        if llm_judge_result and isinstance(llm_judge_result, dict):
                            methods_llm_judge_data.setdefault(method, []).append(
                                llm_judge_result
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

                # Add LLM judge aggregation
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

                # Add token usage aggregation
                if method in methods_token_data:
                    token_data_list = methods_token_data[method]
                    token_agg = _aggregate_token_usage_for_method(
                        method, token_data_list
                    )
                    if token_agg:
                        agg["token_usage"] = token_agg

                # Add timing aggregation
                if method in methods_timing_data:
                    timing_data = methods_timing_data[method]
                    agg["avg_generation_time"] = sum(timing_data) / len(timing_data)
                    agg["total_generation_time"] = sum(timing_data)

                if agg:
                    aggregates[method] = agg
            if aggregates:
                data["evaluation_aggregate"] = aggregates
        except Exception as e:
            logger.warning(f"Failed to compute aggregation: {e}")

        # Save updated results
        logger.info("Saving updated results...")
        save_results(results_dir, data)

        # Generate aggregation summary
        _generate_aggregation_summary(data)

        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Evaluation completed in {elapsed_time:.1f}s")
        logger.info(
            f"📊 New evaluations: {total_evaluations} | successes: {successful_evaluations} | "
            f"skipped_existing: {skipped_existing} | skipped_missing: {skipped_missing}"
        )

        if successful_evaluations < total_evaluations:
            logger.warning(
                f"⚠️ {total_evaluations - successful_evaluations} evaluations failed"
            )

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
