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

def create_results_directory(args):
    """Create an expressive results directory name based on experiment parameters."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build descriptive directory name
    dir_parts = []
    
    # Add method information
    if args.method == "all":
        dir_parts.append("all_methods")
    else:
        dir_parts.append(f"{args.method}_only")
    
    # Add topic count
    dir_parts.append(f"{args.num_topics}topics")
    
    # Add evaluation status
    if args.skip_evaluation:
        dir_parts.append("no_eval")
    else:
        dir_parts.append("with_eval")
    
    # Add timestamp
    dir_parts.append(timestamp)
    
    # Combine into expressive directory name
    dir_name = "_".join(dir_parts)
    
    return Path("results") / dir_name

def main():
    parser = argparse.ArgumentParser(description="AI Writer Framework - Content Generation and Evaluation")
    parser.add_argument("--method", type=str, choices=[
        "direct", "writer_only", "rag", "all"
    ], default="all", help="Generation method(s) to run")
    parser.add_argument("--num_topics", type=int, default=5,
                       help="Number of topics to process (default: 5)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Generate content without evaluation metrics")

    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create results directory
    results_dir = create_results_directory(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = Config()
    
    # Get topics to process
    topics_to_process = get_topics_to_process(args, logger)
    if not topics_to_process:
        logger.error("No topics found to process. Exiting.")
        sys.exit(1)
    
    # Initialize workflows
    workflows = initialize_workflows(args, config)
    
    # Initialize evaluator if needed
    evaluator = None
    evaluation_enabled = not args.skip_evaluation
    if evaluation_enabled:
        freshwiki_loader = FreshWikiLoader()
        eval_stats = freshwiki_loader.validate_evaluation_dataset()
        if eval_stats['status'] == 'no_data':
            logger.warning("No FreshWiki evaluation data found. Running without evaluation.")
            evaluation_enabled = False
        else:
            evaluator = ArticleEvaluator()
            logger.info(f"Evaluation enabled with {eval_stats['total_entries']} reference entries")
    
    logger.info(f"Processing {len(topics_to_process)} topics with {len(workflows)} methods")
    
    # Process topics
    all_results = {}
    
    for topic_idx, topic_data in enumerate(topics_to_process, 1):
        topic = topic_data['topic']
        reference_entry = topic_data.get('reference')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TOPIC {topic_idx}/{len(topics_to_process)}: {topic}")
        logger.info(f"{'='*80}")
        
        topic_results = process_topic(topic, workflows, evaluator, reference_entry, logger)
        all_results[topic] = topic_results
    
    # Save results and generate summary
    save_results_and_summary(all_results, results_dir, evaluation_enabled, logger)

def get_topics_to_process(args, logger):
    """Get topics to process from FreshWiki dataset."""
    freshwiki_loader = FreshWikiLoader()
    eval_stats = freshwiki_loader.validate_evaluation_dataset()
    
    if eval_stats['status'] == 'no_data':
        logger.error("CRITICAL: No FreshWiki data found!")
        logger.error("Please ensure FreshWiki dataset is properly installed at data/freshwiki/")
        logger.error("Expected files: freshwiki.json, freshwiki_data.json, evaluation_set.json, or similar")
        raise FileNotFoundError("FreshWiki evaluation dataset is required but not found")
    
    # Get sample from FreshWiki
    sample_entries = freshwiki_loader.get_evaluation_sample(args.num_topics)
    logger.info(f"Loaded {len(sample_entries)} topics from FreshWiki")
    
    return [{'topic': entry.topic, 'reference': entry} for entry in sample_entries]

def initialize_workflows(args, config):
    """Initialize the requested workflows."""
    workflow_classes = {
        "direct": DirectPromptingWorkflow,
        "writer_only": WriterOnlyWorkflow,
        "rag": RAGWriterWorkflow
    }
    
    workflows = {}
    
    if args.method == "all":
        for name, cls in workflow_classes.items():
            workflows[name] = cls(config.config)
    else:
        workflows[args.method] = workflow_classes[args.method](config.config)
    
    return workflows

def process_topic(topic, workflows, evaluator, reference_entry, logger):
    """Process a single topic with all workflows."""
    topic_results = {
        'generation_results': {},
        'evaluation_results': {},
        'has_reference': reference_entry is not None
    }
    
    for method_name, workflow in workflows.items():
        logger.info(f"\n--- Running {method_name} for '{topic}' ---")
        
        try:
            # Generate content
            article = workflow.generate_content(topic)
            
            # Store generation results
            topic_results['generation_results'][method_name] = {
                'article': {
                    'title': article.title,
                    'content': article.content,
                    'sections': article.sections,
                    'metadata': article.metadata
                },
                'success': True
            }
            
            # Evaluate if possible
            if evaluator and reference_entry:
                metrics = evaluator.evaluate_article(article, reference_entry)
                topic_results['evaluation_results'][method_name] = metrics
                
                logger.info(f"✓ {method_name} completed and evaluated")
                logger.info(f"  Word count: {article.metadata.get('word_count', 'N/A')}")
                logger.info(f"  Key metrics: {format_metrics_summary(metrics)}")
            else:
                logger.info(f"✓ {method_name} completed (no evaluation)")
                logger.info(f"  Word count: {article.metadata.get('word_count', 'N/A')}")
            
        except Exception as e:
            logger.error(f"✗ {method_name} failed for '{topic}': {e}")
            topic_results['generation_results'][method_name] = {
                'error': str(e),
                'success': False
            }
    
    return topic_results

def format_metrics_summary(metrics):
    """Format key metrics for logging."""
    key_metrics = ['word_overlap', 'section_coverage', 'heading_similarity']
    formatted = []
    
    for metric in key_metrics:
        if metric in metrics:
            formatted.append(f"{metric}={metrics[metric]:.3f}")
    
    return ', '.join(formatted) if formatted else 'No metrics available'

def save_results_and_summary(all_results, results_dir, evaluation_enabled, logger):
    """Save comprehensive results and generate summary."""
    # Save detailed results
    results_file = results_dir / "detailed_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Generate summary
    summary = generate_summary(all_results, evaluation_enabled)
    
    # Save summary
    summary_file = results_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    # Display summary
    display_summary(summary, logger)
    
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"  Detailed results: {results_file}")
    logger.info(f"  Summary: {summary_file}")

def generate_summary(all_results, evaluation_enabled):
    """Generate summary statistics from results."""
    method_stats = {}
    
    for topic, topic_data in all_results.items():
        gen_results = topic_data['generation_results']
        eval_results = topic_data['evaluation_results']
        
        for method, result in gen_results.items():
            if method not in method_stats:
                method_stats[method] = {
                    'successes': 0,
                    'failures': 0,
                    'word_counts': [],
                    'evaluation_scores': []
                }
            
            if result.get('success', False):
                method_stats[method]['successes'] += 1
                word_count = result['article']['metadata'].get('word_count', 0)
                method_stats[method]['word_counts'].append(word_count)
                
                if method in eval_results:
                    method_stats[method]['evaluation_scores'].append(eval_results[method])
            else:
                method_stats[method]['failures'] += 1
    
    # Calculate summary statistics
    summary = {
        'total_topics': len(all_results),
        'evaluation_enabled': evaluation_enabled,
        'method_statistics': {}
    }
    
    for method, stats in method_stats.items():
        method_summary = {
            'successes': stats['successes'],
            'failures': stats['failures'],
            'success_rate': stats['successes'] / (stats['successes'] + stats['failures']) if (stats['successes'] + stats['failures']) > 0 else 0,
            'avg_word_count': sum(stats['word_counts']) / len(stats['word_counts']) if stats['word_counts'] else 0
        }
        
        if evaluation_enabled and stats['evaluation_scores']:
            # Calculate average metrics
            all_metrics = set()
            for score_dict in stats['evaluation_scores']:
                all_metrics.update(score_dict.keys())
            
            avg_scores = {}
            for metric in all_metrics:
                values = [score_dict.get(metric, 0) for score_dict in stats['evaluation_scores']]
                avg_scores[metric] = sum(values) / len(values)
            
            method_summary['evaluation_averages'] = avg_scores
        
        summary['method_statistics'][method] = method_summary
    
    return summary

def display_summary(summary, logger):
    """Display summary to console."""
    logger.info(f"\n{'='*80}")
    logger.info("EXECUTION SUMMARY")
    logger.info(f"{'='*80}")
    
    logger.info(f"Total topics processed: {summary['total_topics']}")
    logger.info(f"Evaluation enabled: {summary['evaluation_enabled']}")
    
    for method, stats in summary['method_statistics'].items():
        logger.info(f"\n{method.upper()} RESULTS:")
        logger.info(f"  Success rate: {stats['success_rate']:.2%} ({stats['successes']}/{stats['successes'] + stats['failures']})")
        logger.info(f"  Average word count: {stats['avg_word_count']:.1f}")
        
        if 'evaluation_averages' in stats:
            logger.info("  Evaluation averages:")
            for metric, value in stats['evaluation_averages'].items():
                logger.info(f"    {metric}: {value:.3f}")

if __name__ == "__main__":
    main()