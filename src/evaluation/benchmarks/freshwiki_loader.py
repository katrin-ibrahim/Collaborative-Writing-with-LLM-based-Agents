# src/evaluation/benchmarks/freshwiki_loader.py - Simplified, evaluation-focused loader
import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FreshWikiEntry:
    """
    Represents a FreshWiki evaluation benchmark entry.
    
    FreshWiki serves as ground truth for evaluation - it provides reference
    articles that our generated content will be compared against to measure
    quality across multiple dimensions.
    """
    topic: str                              # The topic/title we're generating content for
    reference_outline: List[str]            # Ground truth section headings for structure evaluation
    reference_content: str                  # Full reference article for content comparison
    ground_truth_sections: Dict[str, str]   # Section-by-section ground truth for detailed analysis
    metadata: Dict[str, Any]               # Additional metadata (dates, categories, etc.)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FreshWikiEntry':
        """
        Create FreshWikiEntry from dictionary data.
        
        This method handles the various formats that FreshWiki data might come in,
        extracting the essential components needed for evaluation while being
        tolerant of different field naming conventions.
        """
        return cls(
            topic=data.get('topic', '') or data.get('title', ''),
            reference_outline=data.get('outline', []) or data.get('headings', []),
            reference_content=data.get('content', '') or data.get('text', ''),
            ground_truth_sections=data.get('sections', {}),
            metadata=data.get('metadata', {})
        )

class FreshWikiLoader:
    """
    Loader for FreshWiki evaluation dataset.
    
    This class has a single, focused responsibility: loading FreshWiki data
    for evaluation purposes. It does NOT provide content for generation -
    that's the job of the SearchEngine and knowledge sources.
    
    The separation of concerns is crucial:
    - Knowledge sources (Wikipedia, web search) → Content generation
    - Evaluation datasets (FreshWiki) → Quality measurement
    """
    
    def __init__(self, data_path: str = "data/freshwiki"):
        self.data_path = Path(data_path)
        self.entries: List[FreshWikiEntry] = []
        self.topic_index: Dict[str, FreshWikiEntry] = {}
        
        # Load dataset on initialization
        self._load_evaluation_dataset()
    
    def _load_evaluation_dataset(self) -> None:
        """
        Load FreshWiki evaluation dataset from JSON files.
        
        This method looks for common FreshWiki file patterns and loads
        them into a standardized format for evaluation use.
        """
        if not self.data_path.exists():
            logger.warning(f"FreshWiki evaluation data not found at: {self.data_path}")
            logger.info("System will run without evaluation benchmarks")
            return
        
        # Common FreshWiki file patterns
        evaluation_files = [
            "freshwiki.json",
            "freshwiki_data.json", 
            "evaluation_set.json",
            "test.json",
            "validation.json"
        ]
        
        for filename in evaluation_files:
            file_path = self.data_path / filename
            if file_path.exists():
                logger.info(f"Loading FreshWiki evaluation data from: {file_path}")
                self._load_from_file(file_path)
                return
        
        # If no standard files found, try any JSON file
        json_files = list(self.data_path.glob("*.json"))
        if json_files:
            logger.info(f"Loading FreshWiki data from: {json_files[0]}")
            self._load_from_file(json_files[0])
        else:
            logger.warning("No FreshWiki evaluation files found - evaluation will be limited")
    
    def _load_from_file(self, file_path: Path) -> None:
        """Load and parse FreshWiki data from a specific JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different data structures
            if isinstance(data, list):
                # List of evaluation entries
                for entry_data in data:
                    entry = self._parse_evaluation_entry(entry_data)
                    if entry and entry.topic:  # Only add valid entries with topics
                        self.entries.append(entry)
                        self.topic_index[entry.topic.lower()] = entry
            
            elif isinstance(data, dict):
                if 'topic' in data or 'title' in data:
                    # Single evaluation entry
                    entry = self._parse_evaluation_entry(data)
                    if entry and entry.topic:
                        self.entries.append(entry)
                        self.topic_index[entry.topic.lower()] = entry
                else:
                    # Dictionary of entries
                    for key, entry_data in data.items():
                        entry = self._parse_evaluation_entry(entry_data)
                        if entry and entry.topic:
                            self.entries.append(entry)
                            self.topic_index[entry.topic.lower()] = entry
            
            logger.info(f"Loaded {len(self.entries)} evaluation entries from FreshWiki")
            
        except Exception as e:
            logger.error(f"Failed to load FreshWiki evaluation data from {file_path}: {e}")
    
    def _parse_evaluation_entry(self, data: Dict[str, Any]) -> Optional[FreshWikiEntry]:
        """
        Parse a single evaluation entry from various possible formats.
        
        This method is tolerant of different field naming conventions
        while extracting the essential components needed for evaluation.
        """
        try:
            # Extract topic/title (required)
            topic = (data.get('topic') or 
                    data.get('title') or 
                    data.get('article_title') or 
                    data.get('name', ''))
            
            if not topic:
                return None  # Skip entries without topics
            
            # Extract reference outline/headings for structure evaluation
            outline = []
            if 'outline' in data:
                outline = data['outline'] if isinstance(data['outline'], list) else []
            elif 'headings' in data:
                outline = data['headings'] if isinstance(data['headings'], list) else []
            elif 'sections' in data and isinstance(data['sections'], list):
                outline = data['sections']
            
            # Extract reference content for content evaluation
            content = (data.get('content') or 
                      data.get('text') or 
                      data.get('article_content') or 
                      data.get('full_text', ''))
            
            # Extract section-wise content for detailed evaluation
            sections = {}
            if 'sections' in data and isinstance(data['sections'], dict):
                sections = data['sections']
            elif 'section_content' in data:
                sections = data['section_content']
            
            # Extract metadata
            metadata = data.get('metadata', {})
            
            return FreshWikiEntry(
                topic=topic,
                reference_outline=outline,
                reference_content=content,
                ground_truth_sections=sections,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse FreshWiki evaluation entry: {e}")
            return None
    
    def get_evaluation_entry(self, topic: str) -> Optional[FreshWikiEntry]:
        """
        Get evaluation ground truth for a specific topic.
        
        This is the primary method used by the evaluation system to get
        reference data for comparing against generated content.
        """
        return self.topic_index.get(topic.lower())
    
    def get_all_evaluation_topics(self) -> List[str]:
        """Get list of all topics available for evaluation."""
        return [entry.topic for entry in self.entries]
    
    def get_evaluation_sample(self, n: int = 5) -> List[FreshWikiEntry]:
        """
        Get a random sample of evaluation entries for benchmarking.
        
        This is useful for running quick evaluation tests or when you want
        to benchmark on a subset of the full evaluation dataset.
        """
        import random
        return random.sample(self.entries, min(n, len(self.entries)))
    
    def validate_evaluation_dataset(self) -> Dict[str, Any]:
        """
        Validate the evaluation dataset and return comprehensive statistics.
        
        This helps you understand the quality and coverage of your evaluation
        data, which is crucial for meaningful benchmarking.
        """
        if not self.entries:
            return {
                'status': 'no_data',
                'message': 'No FreshWiki evaluation data loaded',
                'recommendation': 'Add FreshWiki JSON files to data/freshwiki/ directory'
            }
        
        stats = {
            'status': 'loaded',
            'total_entries': len(self.entries),
            'topics_with_outlines': 0,
            'topics_with_content': 0,
            'topics_with_sections': 0,
            'average_outline_length': 0,
            'average_content_length': 0,
            'content_length_distribution': {'short': 0, 'medium': 0, 'long': 0},
            'sample_topics': []
        }
        
        outline_lengths = []
        content_lengths = []
        
        for entry in self.entries:
            # Count entries with different types of ground truth data
            if entry.reference_outline:
                stats['topics_with_outlines'] += 1
                outline_lengths.append(len(entry.reference_outline))
            
            if entry.reference_content:
                stats['topics_with_content'] += 1
                content_length = len(entry.reference_content)
                content_lengths.append(content_length)
                
                # Categorize content length for distribution analysis
                if content_length < 1000:
                    stats['content_length_distribution']['short'] += 1
                elif content_length < 3000:
                    stats['content_length_distribution']['medium'] += 1
                else:
                    stats['content_length_distribution']['long'] += 1
            
            if entry.ground_truth_sections:
                stats['topics_with_sections'] += 1
            
            # Collect sample topics for manual inspection
            if len(stats['sample_topics']) < 5:
                stats['sample_topics'].append(entry.topic)
        
        # Calculate averages
        if outline_lengths:
            stats['average_outline_length'] = sum(outline_lengths) / len(outline_lengths)
        
        if content_lengths:
            stats['average_content_length'] = sum(content_lengths) / len(content_lengths)
        
        # Add data quality assessment
        stats['data_quality'] = {
            'outline_coverage': stats['topics_with_outlines'] / stats['total_entries'],
            'content_coverage': stats['topics_with_content'] / stats['total_entries'],
            'section_coverage': stats['topics_with_sections'] / stats['total_entries']
        }
        
        return stats


# Updated main.py - Clean separation of concerns
import argparse
import logging
import json
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from workflows.direct_prompting import DirectPromptingWorkflow
from workflows.writer_only import WriterOnlyWorkflow
from workflows.rag_writer import RAGWriterWorkflow
from workflows.co_storm_enhanced import COStormEnhancedWorkflow
from evaluation.benchmarks.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.config import Config
from utils.logging_setup import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Content Generation System with Clean Architecture")
    parser.add_argument("--topic", type=str, help="Topic to generate content for")
    parser.add_argument("--method", type=str, choices=[
        "direct", "writer_only", "rag", "co_storm"
    ], default="co_storm", help="Generation method to use")
    parser.add_argument("--freshwiki_path", type=str, default="data/freshwiki",
                       help="Path to FreshWiki evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate against FreshWiki ground truth (if available)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark comparing all methods")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = Config()
    
    # Initialize evaluation dataset (separate from knowledge sources)
    evaluation_loader = FreshWikiLoader(args.freshwiki_path)
    
    # Validate evaluation dataset
    eval_stats = evaluation_loader.validate_evaluation_dataset()
    logger.info("FreshWiki Evaluation Dataset Status:")
    if eval_stats['status'] == 'no_data':
        logger.warning(f"  {eval_stats['message']}")
        logger.info(f"  {eval_stats['recommendation']}")
    else:
        logger.info(f"  Total evaluation entries: {eval_stats['total_entries']}")
        logger.info(f"  Entries with content: {eval_stats['topics_with_content']}")
        logger.info(f"  Entries with outlines: {eval_stats['topics_with_outlines']}")
        logger.info(f"  Content coverage: {eval_stats['data_quality']['content_coverage']:.1%}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.benchmark:
        run_comprehensive_benchmark(config, evaluation_loader, output_dir, logger)
    else:
        run_single_generation(args, config, evaluation_loader, output_dir, logger)

def run_single_generation(args, config, evaluation_loader, output_dir, logger):
    """
    Run content generation for a single topic.
    
    This demonstrates the clean separation:
    - Generation uses knowledge sources (Wikipedia, web search)  
    - Evaluation uses FreshWiki ground truth (if available)
    """
    
    # Determine topic and get evaluation reference (if available)
    if args.topic:
        topic = args.topic
        evaluation_reference = evaluation_loader.get_evaluation_entry(topic)
    else:
        # Use topic from evaluation dataset if available, otherwise default
        if evaluation_loader.entries:
            evaluation_reference = evaluation_loader.get_evaluation_sample(1)[0]
            topic = evaluation_reference.topic
        else:
            topic = "Artificial Intelligence"  # Default topic
            evaluation_reference = None
    
    logger.info(f"Generating content for topic: {topic}")
    if evaluation_reference:
        logger.info("Evaluation reference available for quality assessment")
    else:
        logger.info("No evaluation reference - will provide basic metrics only")
    
    # Initialize content generation workflow
    workflows = {
        "direct": DirectPromptingWorkflow,
        "writer_only": WriterOnlyWorkflow, 
        "rag": RAGWriterWorkflow,
        "co_storm": COStormEnhancedWorkflow
    }
    
    workflow_class = workflows[args.method]
    workflow = workflow_class(config.config)
    
    # Generate content using knowledge sources (NOT FreshWiki)
    try:
        logger.info(f"Running {args.method} workflow...")
        article = workflow.generate_content(topic)
        
        # Save generation results
        result = {
            "topic": topic,
            "method": args.method,
            "generation_timestamp": str(Path().cwd()),
            "article": {
                "title": article.title,
                "content": article.content,
                "sections": article.sections,
                "metadata": article.metadata
            }
        }
        
        # Save to file
        output_file = output_dir / f"{args.method}_{topic.replace(' ', '_')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generation results saved to: {output_file}")
        
        # Run evaluation if requested AND evaluation reference is available
        if args.evaluate and evaluation_reference:
            evaluator = ArticleEvaluator()
            evaluation_result = evaluator.evaluate_article(article, evaluation_reference)
            
            logger.info("Evaluation Results (compared to FreshWiki ground truth):")
            for metric, score in evaluation_result.items():
                logger.info(f"  {metric}: {score:.3f}")
            
            # Save evaluation results
            eval_file = output_dir / f"eval_{args.method}_{topic.replace(' ', '_')}.json"
            evaluation_data = {
                "topic": topic,
                "method": args.method,
                "evaluation_metrics": evaluation_result,
                "reference_source": "freshwiki"
            }
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2)
                
            logger.info(f"Evaluation results saved to: {eval_file}")
        
        elif args.evaluate and not evaluation_reference:
            logger.warning("Evaluation requested but no FreshWiki reference found for this topic")
        
        # Print generation summary
        logger.info(f"\nGeneration Summary:")
        logger.info(f"  Title: {article.title}")
        logger.info(f"  Word Count: {article.metadata.get('word_count', 'N/A')}")
        logger.info(f"  Sections: {len(article.sections) if article.sections else 0}")
        logger.info(f"  Method: {args.method}")
        logger.info(f"  Knowledge Sources: {article.metadata.get('knowledge_sources', 'LLM internal knowledge')}")
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}", exc_info=True)

def run_comprehensive_benchmark(config, evaluation_loader, output_dir, logger):
    """
    Run comprehensive benchmark comparing all generation methods.
    
    This function demonstrates the proper use of evaluation data:
    - Each method generates content using its own knowledge sources
    - All generated content is then evaluated against FreshWiki ground truth
    - Results are compared across methods for systematic analysis
    """
    
    logger.info("Running comprehensive benchmark across all methods...")
    
    # Determine test topics
    if evaluation_loader.entries:
        test_entries = evaluation_loader.get_evaluation_sample(5)
        test_topics = [(entry.topic, entry) for entry in test_entries]
        logger.info(f"Using {len(test_topics)} topics from FreshWiki evaluation set")
    else:
        # Use default topics if no evaluation data available
        default_topics = [
            "Artificial Intelligence", "Climate Change", "Quantum Computing",
            "Renewable Energy", "Space Exploration"
        ]
        test_topics = [(topic, None) for topic in default_topics]
        logger.info(f"No FreshWiki data available, using {len(test_topics)} default topics")
    
    # Initialize all generation workflows
    workflows = {
        "direct": DirectPromptingWorkflow(config.config),
        "writer_only": WriterOnlyWorkflow(config.config),
        "rag": RAGWriterWorkflow(config.config),
        "co_storm": COStormEnhancedWorkflow(config.config)
    }
    
    benchmark_results = {}
    evaluator = ArticleEvaluator()
    
    # Run each method on each topic
    for topic, evaluation_reference in test_topics:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking topic: {topic}")
        logger.info(f"{'='*60}")
        
        topic_results = {}
        
        for method_name, workflow in workflows.items():
            try:
                logger.info(f"  Running {method_name}...")
                
                # Generate content using the method's knowledge sources
                article = workflow.generate_content(topic)
                
                # Collect basic metrics
                basic_metrics = {
                    "word_count": article.metadata.get("word_count", 0),
                    "sections_count": len(article.sections) if article.sections else 0,
                    "title": article.title,
                    "generation_method": method_name
                }
                
                # Evaluate against FreshWiki ground truth (if available)
                evaluation_metrics = {}
                if evaluation_reference:
                    evaluation_metrics = evaluator.evaluate_article(article, evaluation_reference)
                    logger.info(f"    Evaluated against FreshWiki reference")
                else:
                    logger.info(f"    No evaluation reference available")
                
                topic_results[method_name] = {
                    "basic_metrics": basic_metrics,
                    "evaluation_metrics": evaluation_metrics,
                    "success": True,
                    "has_evaluation_reference": evaluation_reference is not None
                }
                
                logger.info(f"    ✓ {method_name} completed: {basic_metrics['word_count']} words")
                
            except Exception as e:
                logger.error(f"    ✗ {method_name} failed: {e}")
                topic_results[method_name] = {
                    "error": str(e),
                    "success": False
                }
        
        benchmark_results[topic] = topic_results
    
    # Save comprehensive benchmark results
    results_file = output_dir / "comprehensive_benchmark.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nBenchmark results saved to: {results_file}")
    
    # Generate and display summary analysis
    generate_benchmark_analysis(benchmark_results, output_dir, logger)

def generate_benchmark_analysis(results, output_dir, logger):
    """Generate comprehensive analysis of benchmark results."""
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE BENCHMARK ANALYSIS")
    logger.info(f"{'='*80}")
    
    # Aggregate statistics by method
    method_statistics = {}
    
    for topic, topic_results in results.items():
        for method, result in topic_results.items():
            if result.get("success", False):
                if method not in method_statistics:
                    method_statistics[method] = {
                        "successful_runs": 0,
                        "word_counts": [],
                        "section_counts": [],
                        "evaluation_scores": []
                    }
                
                stats = method_statistics[method]
                stats["successful_runs"] += 1
                
                basic = result["basic_metrics"]
                stats["word_counts"].append(basic["word_count"])
                stats["section_counts"].append(basic["sections_count"])
                
                # Collect evaluation scores if available
                eval_metrics = result.get("evaluation_metrics", {})
                if eval_metrics and result.get("has_evaluation_reference", False):
                    stats["evaluation_scores"].append(eval_metrics)
    
    # Calculate and display summary statistics
    analysis_summary = {}
    
    for method, stats in method_statistics.items():
        # Basic statistics
        avg_words = sum(stats["word_counts"]) / len(stats["word_counts"]) if stats["word_counts"] else 0
        avg_sections = sum(stats["section_counts"]) / len(stats["section_counts"]) if stats["section_counts"] else 0
        
        method_summary = {
            "successful_runs": stats["successful_runs"],
            "average_word_count": round(avg_words, 1),
            "average_sections": round(avg_sections, 1),
            "word_count_range": [min(stats["word_counts"]), max(stats["word_counts"])] if stats["word_counts"] else [0, 0]
        }
        
        # Evaluation metrics (if available)
        if stats["evaluation_scores"]:
            evaluation_summary = {}
            all_metrics = set()
            
            # Collect all metric names
            for eval_result in stats["evaluation_scores"]:
                all_metrics.update(eval_result.keys())
            
            # Calculate averages for each metric
            for metric in all_metrics:
                values = [eval_result.get(metric, 0) for eval_result in stats["evaluation_scores"]]
                evaluation_summary[metric] = round(sum(values) / len(values), 3)
            
            method_summary["evaluation_averages"] = evaluation_summary
            method_summary["evaluated_topics"] = len(stats["evaluation_scores"])
        
        analysis_summary[method] = method_summary
        
        # Display method summary
        logger.info(f"\n{method.upper()} PERFORMANCE:")
        logger.info(f"  Successful runs: {method_summary['successful_runs']}")
        logger.info(f"  Average word count: {method_summary['average_word_count']}")
        logger.info(f"  Average sections: {method_summary['average_sections']}")
        logger.info(f"  Word count range: {method_summary['word_count_range'][0]}-{method_summary['word_count_range'][1]}")
        
        if "evaluation_averages" in method_summary:
            logger.info(f"  Evaluated topics: {method_summary['evaluated_topics']}")
            logger.info("  Average evaluation scores:")
            for metric, score in method_summary["evaluation_averages"].items():
                logger.info(f"    {metric}: {score}")
        else:
            logger.info("  No evaluation scores available (no FreshWiki references)")
    
    # Save analysis summary
    analysis_file = output_dir / "benchmark_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nDetailed analysis saved to: {analysis_file}")
    
    # Method comparison (if evaluation data available)
    methods_with_eval = {k: v for k, v in analysis_summary.items() 
                        if "evaluation_averages" in v}
    
    if methods_with_eval:
        logger.info(f"\n{'='*60}")
        logger.info("METHOD COMPARISON (Based on FreshWiki Evaluation)")
        logger.info(f"{'='*60}")
        
        # Compare key metrics
        comparison_metrics = ["rouge_1", "rouge_2", "rouge_l", "heading_soft_recall"]
        
        for metric in comparison_metrics:
            logger.info(f"\n{metric.upper()} RANKING:")
            method_scores = []
            for method, summary in methods_with_eval.items():
                score = summary["evaluation_averages"].get(metric, 0)
                method_scores.append((method, score))
            
            # Sort by score (descending)
            method_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (method, score) in enumerate(method_scores, 1):
                logger.info(f"  {i}. {method}: {score:.3f}")

if __name__ == "__main__":
    main()