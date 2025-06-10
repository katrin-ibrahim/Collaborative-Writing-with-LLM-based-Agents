import argparse
import logging
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.benchmarks.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.logging_setup import setup_logging
from utils.data_models import Article

def create_results_directory(args):
    """Create results directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_parts = ["storm", args.method, f"{args.num_topics}topics", timestamp]
    return Path("results") / "_".join(dir_parts)

def initialize_storm(args):
    """Initialize STORM exactly as intended by the package."""
    try:
        from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
        from knowledge_storm.lm import LitellmModel  # Correct import for 1.1.0
        from knowledge_storm.rm import DuckDuckGoSearchRM
        
        logger = logging.getLogger(__name__)
        
        # Check HF token
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logger.error("HF_TOKEN environment variable not set")
            logger.error("Get token at: https://huggingface.co/settings/tokens")
            return None
        
        # Use STORM's LM configs as designed
        lm_config = STORMWikiLMConfigs()
        
        # Create LiteLLM instance for HuggingFace (correct API for 1.1.0)
        openai_kwargs = {
            'api_key': os.getenv('GROQ_API_KEY', hf_token),  # Use GROQ API key if available
            'temperature': 0.7,
            'top_p': 0.9,
        }
        
        llm = LitellmModel(
            model=args.hf_model,
            max_tokens=100,
            **openai_kwargs
        )
        
        # Set all STORM LM components to use the same model (correct methods for 1.1.0)
        lm_config.set_conv_simulator_lm(llm)
        lm_config.set_question_asker_lm(llm)
        lm_config.set_outline_gen_lm(llm)
        lm_config.set_article_gen_lm(llm)
        lm_config.set_article_polish_lm(llm)
        
        # Use DuckDuckGo search (no API key needed)
        rm = DuckDuckGoSearchRM(k=3)
        
        # Create STORM runner arguments (required for 1.1.0)
        engine_args = STORMWikiRunnerArguments(
            output_dir="./storm_output",
            max_conv_turn=3,
            max_perspective=2,
            search_top_k=3,
            max_thread_num=1  # Reduce for stability with free APIs
        )
        
        # Create STORM runner with correct arguments order for 1.1.0
        storm_runner = STORMWikiRunner(engine_args, lm_config, rm)
        
        logger.info(f"STORM 1.1.0 initialized with {args.hf_model}")
        return storm_runner
        
    except ImportError as e:
        logger.error(f"Install STORM: pip install knowledge-storm==1.1.0 litellm duckduckgo-search")
        return None
    except Exception as e:
        logger.error(f"STORM initialization failed: {e}")
        return None

def run_storm_on_topic(storm_runner, topic, output_base_dir):
    """Run STORM pipeline on a topic exactly as designed."""
    try:
        # Create topic-specific output directory
        topic_dir = output_base_dir / topic.replace(" ", "_").replace("/", "_")
        topic_dir.mkdir(parents=True, exist_ok=True)
        
        # Run STORM pipeline as intended (correct API for 1.1.0 - no output_dir parameter)
        storm_runner.run(
            topic=topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True
        )
        
        # STORM 1.1.0 saves to the output_dir specified in STORMWikiRunnerArguments
        # Look for files in the STORM's configured output directory
        storm_output_dir = Path(storm_runner.args.output_dir)
        topic_subdir = storm_output_dir / topic.replace(" ", "_").replace("/", "_")
        
        # Read the generated article
        article_file = topic_subdir / "storm_gen_article_polished.txt"
        if article_file.exists():
            content = article_file.read_text(encoding='utf-8')
        else:
            # Fallback to unpolished article
            article_file = topic_subdir / "storm_gen_article.txt"
            if article_file.exists():
                content = article_file.read_text(encoding='utf-8')
            else:
                content = f"# {topic}\n\nSTORM completed but no article file found in {topic_subdir}"
        
        return Article(
            title=topic,
            content=content,
            sections={},  # STORM handles sectioning internally
            metadata={
                "method": "storm_full",
                "word_count": len(content.split()),
                "output_dir": str(topic_subdir)
            }
        ), None
        
    except Exception as e:
        return Article(
            title=topic,
            content=f"# {topic}\n\nSTORM Error: {e}",
            sections={},
            metadata={"error": str(e)}
        ), str(e)

def create_direct_baseline(storm_runner, topic):
    """Create direct prompting baseline using STORM's LLM."""
    try:
        # Get STORM's article generation LLM (correct attribute name for 1.1.0)
        llm = storm_runner.lm_configs.article_gen_lm
        
        # Simple direct prompt (let STORM's LLM handle the details)
        prompt = f"Write a comprehensive Wikipedia-style article about {topic}. Include multiple sections with proper headings and detailed content."
        
        # Use STORM's LLM directly (correct API for LitellmModel in 1.1.0)
        response = llm(prompt, max_tokens=100)
        content = response if isinstance(response, str) else str(response)
        
        return Article(
            title=topic,
            content=content,
            sections={},
            metadata={
                "method": "direct_prompting",
                "word_count": len(content.split())
            }
        ), None
        
    except Exception as e:
        return Article(
            title=topic,
            content=f"# {topic}\n\nDirect prompting error: {e}",
            sections={},
            metadata={"error": str(e)}
        ), str(e)

def main():
    parser = argparse.ArgumentParser(description="STORM Baselines - Use As-Is")
    parser.add_argument("--method", choices=["direct", "storm", "all"], default="all")
    parser.add_argument("--num_topics", type=int, default=3)
    parser.add_argument("--hf_model", default="groq/llama-3.1-8b-instant",
                       help="HuggingFace model via LiteLLM")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🌩️  STORM Baselines - Using STORM As-Is")
    logger.info(f"Model: {args.hf_model}")
    logger.info(f"Methods: {args.method}")
    
    # Create results directory
    results_dir = create_results_directory(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize STORM
    storm_runner = initialize_storm(args)
    if not storm_runner:
        logger.error("Failed to initialize STORM")
        return
    
    # Get topics from FreshWiki
    logger.info("Loading topics from FreshWiki...")
    freshwiki_loader = FreshWikiLoader()
    sample_entries = freshwiki_loader.get_evaluation_sample(args.num_topics)
    
    if not sample_entries:
        logger.error("No FreshWiki entries found")
        return
    
    logger.info(f"Loaded {len(sample_entries)} topics")
    
    # Initialize evaluator
    evaluator = None
    if not args.skip_evaluation:
        evaluator = ArticleEvaluator()
        logger.info("Evaluation enabled")
    
    # Create STORM output directory
    storm_output_dir = results_dir / "storm_outputs"
    storm_output_dir.mkdir(exist_ok=True)
    
    # Process topics
    all_results = {}
    
    for i, entry in enumerate(sample_entries, 1):
        topic = entry.topic
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {i}/{len(sample_entries)}: {topic}")
        logger.info(f"{'='*60}")
        
        topic_results = {"generation_results": {}, "evaluation_results": {}}
        
        # Run direct prompting baseline
        if args.method in ["direct", "all"]:
            logger.info("🔄 Running direct prompting baseline...")
            article, error = create_direct_baseline(storm_runner, topic)
            
            topic_results["generation_results"]["direct"] = {
                "success": error is None,
                "word_count": article.metadata.get("word_count", 0),
                "error": error
            }
            
            if evaluator and error is None:
                metrics = evaluator.evaluate_article(article, entry)
                topic_results["evaluation_results"]["direct"] = metrics
                logger.info(f"✓ Direct prompting: {article.metadata.get('word_count', 0)} words")
            elif error:
                logger.error(f"✗ Direct prompting failed: {error}")
        
        # Run full STORM pipeline
        if args.method in ["storm", "all"]:
            logger.info("🔄 Running full STORM pipeline...")
            article, error = run_storm_on_topic(storm_runner, topic, storm_output_dir)
            
            topic_results["generation_results"]["storm"] = {
                "success": error is None,
                "word_count": article.metadata.get("word_count", 0),
                "output_dir": article.metadata.get("output_dir"),
                "error": error
            }
            
            if evaluator and error is None:
                metrics = evaluator.evaluate_article(article, entry)
                topic_results["evaluation_results"]["storm"] = metrics
                logger.info(f"✓ STORM: {article.metadata.get('word_count', 0)} words")
            elif error:
                logger.error(f"✗ STORM failed: {error}")
        
        all_results[topic] = topic_results
    
    # Save results
    results_file = results_dir / "storm_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    methods_run = []
    if args.method in ["direct", "all"]:
        methods_run.append("direct")
    if args.method in ["storm", "all"]:
        methods_run.append("storm")
    
    for method in methods_run:
        successes = sum(1 for r in all_results.values() 
                       if r["generation_results"].get(method, {}).get("success", False))
        total_words = sum(r["generation_results"].get(method, {}).get("word_count", 0)
                         for r in all_results.values()
                         if r["generation_results"].get(method, {}).get("success", False))
        avg_words = total_words / max(successes, 1)
        
        logger.info(f"{method.upper()}: {successes}/{len(all_results)} successful, avg {avg_words:.0f} words")
        
        if evaluator and successes > 0:
            # Calculate average metrics
            rouge_scores = []
            for result in all_results.values():
                if method in result["evaluation_results"]:
                    rouge_scores.append(result["evaluation_results"][method].get("rouge_1", 0))
            
            if rouge_scores:
                avg_rouge = sum(rouge_scores) / len(rouge_scores)
                logger.info(f"  Average ROUGE-1: {avg_rouge:.3f}")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"STORM outputs in: {storm_output_dir}")

if __name__ == "__main__":
    main()