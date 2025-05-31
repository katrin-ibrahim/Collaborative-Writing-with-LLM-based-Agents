import argparse
import logging
import json
from pathlib import Path
import sys
from datetime import datetime

from workflows.direct_prompting import DirectPromptingWorkflow
from workflows.writer_only import WriterOnlyWorkflow
from workflows.rag_writer import RAGWriterWorkflow
from evaluation.benchmarks.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.config import Config
from utils.logging_setup import setup_logging

def main():
    parser = argparse.ArgumentParser(description="AI Writer Framework - Baseline Evaluation")
    parser.add_argument("--method", type=str, choices=[
        "direct", "writer_only", "rag", "all"
    ], default="all", help="Generation method(s) to evaluate")
    parser.add_argument("--num_topics", type=int, default=5,
                       help="Number of topics to evaluate (default: 5)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation, just generate content")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"evaluation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = Config()
    
    # Initialize FreshWiki loader for topics and evaluation
    freshwiki_loader = FreshWikiLoader() 
    
    # Get evaluation topics
    eval_stats = freshwiki_loader.validate_evaluation_dataset()
    if eval_stats['status'] == 'no_data':
        logger.error("No FreshWiki data found.")
        can_evaluate = False
    else:
        logger.info(f"Found {eval_stats['total_entries']} FreshWiki entries")
        test_entries = freshwiki_loader.get_evaluation_sample(args.num_topics)
        test_topics = [(entry.topic, entry) for entry in test_entries]
        can_evaluate = True and not args.skip_evaluation
    
    # Initialize workflows
    workflows = {}
    if args.method == "all":
        workflows = {
            "direct": DirectPromptingWorkflow(config.config),
            "writer_only": WriterOnlyWorkflow(config.config),
            "rag": RAGWriterWorkflow(config.config)
        }
    else:
        workflow_classes = {
            "direct": DirectPromptingWorkflow,
            "writer_only": WriterOnlyWorkflow,
            "rag": RAGWriterWorkflow
        }
        workflows[args.method] = workflow_classes[args.method](config.config)
    
    # Initialize evaluator if we can evaluate
    evaluator = ArticleEvaluator() if can_evaluate else None
    
    logger.info(f"Starting evaluation with {len(test_topics)} topics and {len(workflows)} methods")
    
    # Results storage
    all_results = {}
    evaluation_summary = {}
    
    # Process each topic
    for topic_idx, (topic, reference_entry) in enumerate(test_topics, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TOPIC {topic_idx}/{len(test_topics)}: {topic}")
        logger.info(f"{'='*80}")
        
        topic_results = {}
        topic_evaluations = {}
        
        # Test each workflow on this topic
        for method_name, workflow in workflows.items():
            logger.info(f"\n--- Running {method_name} for '{topic}' ---")
            
            try:
                # Generate content
                article = workflow.generate_content(topic)
                
                # Store generation results
                topic_results[method_name] = {
                    "article": {
                        "title": article.title,
                        "content": article.content,
                        "sections": article.sections,
                        "metadata": article.metadata
                    },
                    "success": True
                }
                
                # Evaluate if possible
                if can_evaluate and reference_entry:
                    metrics = evaluator.evaluate_article(article, reference_entry)
                    topic_evaluations[method_name] = metrics
                    
                    logger.info(f"✓ {method_name} completed and evaluated")
                    logger.info(f"  Word count: {article.metadata.get('word_count', 'N/A')}")
                    logger.info(f"  Key metrics: {', '.join([f'{k}={v:.3f}' for k, v in list(metrics.items())[:3]])}")
                else:
                    logger.info(f"✓ {method_name} completed (no evaluation)")
                    logger.info(f"  Word count: {article.metadata.get('word_count', 'N/A')}")
                
            except Exception as e:
                logger.error(f"✗ {method_name} failed for '{topic}': {e}")
                topic_results[method_name] = {
                    "error": str(e),
                    "success": False
                }
        
        # Store results for this topic
        all_results[topic] = {
            "generation_results": topic_results,
            "evaluation_results": topic_evaluations,
            "has_reference": reference_entry is not None
        }
    
    # Save comprehensive results
    results_file = results_dir / "comprehensive_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    # Generate and display summary
    generate_evaluation_summary(all_results, results_dir, logger, can_evaluate)

def generate_evaluation_summary(results, results_dir, logger, can_evaluate):
    """Generate and display comprehensive evaluation summary."""
    
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*80}")
    
    # Count successes and failures
    method_stats = {}
    
    for topic, topic_data in results.items():
        gen_results = topic_data["generation_results"]
        eval_results = topic_data["evaluation_results"]
        
        for method, result in gen_results.items():
            if method not in method_stats:
                method_stats[method] = {
                    "successes": 0,
                    "failures": 0,
                    "word_counts": [],
                    "evaluation_scores": []
                }
            
            if result.get("success", False):
                method_stats[method]["successes"] += 1
                word_count = result["article"]["metadata"].get("word_count", 0)
                method_stats[method]["word_counts"].append(word_count)
                
                # Collect evaluation scores if available
                if method in eval_results:
                    method_stats[method]["evaluation_scores"].append(eval_results[method])
            else:
                method_stats[method]["failures"] += 1
    
    # Display method comparison
    summary_data = {}
    
    for method, stats in method_stats.items():
        logger.info(f"\n{method.upper()} RESULTS:")
        logger.info(f"  Successful generations: {stats['successes']}")
        logger.info(f"  Failed generations: {stats['failures']}")
        
        if stats["word_counts"]:
            avg_words = sum(stats["word_counts"]) / len(stats["word_counts"])
            logger.info(f"  Average word count: {avg_words:.1f}")
            logger.info(f"  Word count range: {min(stats['word_counts'])}-{max(stats['word_counts'])}")
        
        method_summary = {
            "successes": stats["successes"],
            "failures": stats["failures"],
            "avg_word_count": sum(stats["word_counts"]) / len(stats["word_counts"]) if stats["word_counts"] else 0
        }
        
        if can_evaluate and stats["evaluation_scores"]:
            logger.info(f"  Evaluated topics: {len(stats['evaluation_scores'])}")
            
            # Calculate average scores across all metrics
            all_metrics = set()
            for score_dict in stats["evaluation_scores"]:
                all_metrics.update(score_dict.keys())
            
            avg_scores = {}
            for metric in all_metrics:
                values = [score_dict.get(metric, 0) for score_dict in stats["evaluation_scores"]]
                avg_scores[metric] = sum(values) / len(values)
                logger.info(f"    {metric}: {avg_scores[metric]:.3f}")
            
            method_summary["evaluation_averages"] = avg_scores
        
        summary_data[method] = method_summary
    
    # Save summary
    summary_file = results_dir / "evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    logger.info(f"\nSummary saved to: {summary_file}")
    
    # Method ranking if we have evaluation data
    if can_evaluate:
        display_method_ranking(summary_data, logger)

def display_method_ranking(summary_data, logger):
    """Display ranking of methods based on evaluation metrics."""
    
    methods_with_eval = {k: v for k, v in summary_data.items() 
                        if "evaluation_averages" in v}
    
    if not methods_with_eval:
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("METHOD RANKING (Based on Evaluation Metrics)")
    logger.info(f"{'='*60}")
    
    # Rank by key metrics
    key_metrics = ["word_overlap", "section_coverage", "heading_similarity"]
    
    for metric in key_metrics:
        logger.info(f"\n{metric.upper()} RANKING:")
        method_scores = []
        
        for method, summary in methods_with_eval.items():
            score = summary["evaluation_averages"].get(metric, 0)
            method_scores.append((method, score))
        
        # Sort by score (descending)
        method_scores.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (method, score) in enumerate(method_scores, 1):
            logger.info(f"  {rank}. {method}: {score:.3f}")

if __name__ == "__main__":
    main()