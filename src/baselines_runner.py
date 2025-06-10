import argparse
import logging
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, environment variables should be set manually

from evaluation.benchmarks.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.logging_setup import setup_logging
from utils.data_models import Article
from config.storm_config import ConfigManager

def create_results_directory(args):
    """Create results directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_parts = ["storm", args.method, f"{args.num_topics}topics", timestamp]
    return Path("results") / "_".join(dir_parts)

def initialize_storm_with_retry(args, config_manager: ConfigManager, max_retries: int = 3) -> Optional[object]:
    """Initialize STORM with retry logic and configurable providers."""
    
    try:
        from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
        from knowledge_storm.lm import LitellmModel
        from knowledge_storm.rm import DuckDuckGoSearchRM
        
        logger = logging.getLogger(__name__)
        
        # Get the best available provider from config
        provider_config = config_manager.get_best_provider(args.provider)
        if not provider_config:
            logger.error("No API providers available! Check your .env file and API keys.")
            return None
        
        logger.info(f"Using {provider_config.name} provider with model: {provider_config.model}")
        
        # Get API key from environment
        api_key = os.getenv(provider_config.api_key_env)
        if not api_key:
            logger.error(f"API key not found for {provider_config.name}. Set {provider_config.api_key_env} in your .env file.")
            return None
        
        for attempt in range(max_retries):
            try:
                # Use STORM's LM configs as designed
                lm_config = STORMWikiLMConfigs()
                
                # Create LiteLLM instance with provider-specific settings
                openai_kwargs = {
                    'api_key': api_key,
                    'temperature': provider_config.temperature,
                    'top_p': 0.9,
                    'timeout': provider_config.timeout
                }
                
                # Add provider-specific API base if needed
                if provider_config.api_base:
                    openai_kwargs['api_base'] = provider_config.api_base
                
                # Create LLM with configured settings
                llm = LitellmModel(
                    model=provider_config.model,
                    max_tokens=provider_config.max_tokens,
                    **openai_kwargs
                )
                
                # Set all STORM LM components (keeping exact same API)
                lm_config.set_conv_simulator_lm(llm)
                lm_config.set_question_asker_lm(llm)
                lm_config.set_outline_gen_lm(llm)
                lm_config.set_article_gen_lm(llm)
                lm_config.set_article_polish_lm(llm)
                
                # Use DuckDuckGo search (no API key needed)
                rm = DuckDuckGoSearchRM(k=config_manager.storm_config.search_top_k)
                
                # Create STORM runner arguments with configured settings
                engine_args = STORMWikiRunnerArguments(
                    output_dir="./storm_output",
                    max_conv_turn=config_manager.storm_config.max_conv_turn,
                    max_perspective=config_manager.storm_config.max_perspective,
                    search_top_k=config_manager.storm_config.search_top_k,
                    max_thread_num=config_manager.storm_config.max_thread_num
                )
                
                # Create STORM runner (keeping exact same API)
                storm_runner = STORMWikiRunner(engine_args, lm_config, rm)
                
                # Test the configuration with a simple call
                logger.info("Testing STORM configuration...")
                test_response = llm("Test", max_tokens=10)
                logger.info(f"✓ STORM initialized successfully with {provider_config.name}")
                
                return storm_runner
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed with {provider_config.name}: {e}")
                
                if attempt < max_retries - 1:
                    # Try next available provider
                    available_providers = list(config_manager.get_available_providers().keys())
                    if len(available_providers) > 1:
                        current_index = available_providers.index(provider_config.name)
                        next_provider_name = available_providers[(current_index + 1) % len(available_providers)]
                        provider_config = config_manager.providers[next_provider_name]
                        api_key = os.getenv(provider_config.api_key_env)
                        
                        logger.info(f"Switching to {provider_config.name} provider")
                    
                    time.sleep(2)  # Brief delay before retry
                else:
                    logger.error(f"All initialization attempts failed")
        
        return None
        
    except ImportError as e:
        logger.error("STORM package not installed. Install with: pip install knowledge-storm==1.1.0 litellm duckduckgo-search")
        return None
    except Exception as e:
        logger.error(f"STORM initialization failed: {e}")
        return None

def run_storm_on_topic_with_retry(storm_runner, topic: str, output_base_dir: Path, 
                                 config_manager: ConfigManager) -> Tuple[Article, Optional[str]]:
    """Run STORM pipeline with retry logic and better error handling."""
    
    logger = logging.getLogger(__name__)
    max_retries = config_manager.storm_config.max_retries
    
    for attempt in range(max_retries):
        try:
            # Create topic-specific output directory
            topic_dir = output_base_dir / topic.replace(" ", "_").replace("/", "_")
            topic_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Running STORM (attempt {attempt + 1}/{max_retries})...")
            
            # Run STORM pipeline (keeping exact same API call)
            storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=config_manager.storm_config.enable_polish
            )
            
            # Read the generated article (keeping same file reading logic)
            storm_output_dir = Path(storm_runner.args.output_dir)
            topic_subdir = storm_output_dir / topic.replace(" ", "_").replace("/", "_")
            
            # Try both polished and unpolished versions
            article_files = [
                topic_subdir / "storm_gen_article_polished.txt",
                topic_subdir / "storm_gen_article.txt"
            ]
            
            content = None
            for article_file in article_files:
                if article_file.exists():
                    content = article_file.read_text(encoding='utf-8')
                    logger.info(f"✓ Read article from {article_file.name}")
                    break
            
            if not content:
                # Check if any files were created
                if topic_subdir.exists():
                    files = list(topic_subdir.glob("*.txt"))
                    if files:
                        content = files[0].read_text(encoding='utf-8')
                        logger.info(f"✓ Read article from {files[0].name}")
                
            if not content:
                content = f"# {topic}\n\nSTORM completed but no article content found in {topic_subdir}"
            
            return Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm_full",
                    "word_count": len(content.split()),
                    "output_dir": str(topic_subdir),
                    "attempts": attempt + 1
                }
            ), None
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"STORM attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                # Wait before retry, with exponential backoff
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                # Final attempt failed
                return Article(
                    title=topic,
                    content=f"# {topic}\n\nSTORM Error after {max_retries} attempts: {error_msg}",
                    sections={},
                    metadata={"error": error_msg, "attempts": max_retries}
                ), error_msg

def create_direct_baseline_with_retry(storm_runner, topic: str, 
                                    config_manager: ConfigManager) -> Tuple[Article, Optional[str]]:
    """Create direct prompting baseline with retry logic."""
    
    logger = logging.getLogger(__name__)
    max_retries = config_manager.storm_config.max_retries
    
    for attempt in range(max_retries):
        try:
            # Get STORM's article generation LLM (same as before)
            llm = storm_runner.lm_configs.article_gen_lm
            
            # Simple direct prompt
            prompt = f"Write a comprehensive Wikipedia-style article about {topic}. Include multiple sections with proper headings and detailed content."
            
            # Use configured max_tokens
            provider_config = config_manager.get_best_provider()
            max_tokens = provider_config.max_tokens if provider_config else 50
            
            # Use STORM's LLM with timeout handling
            response = llm(prompt, max_tokens=max_tokens)
            content = response if isinstance(response, str) else str(response)
            
            return Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct_prompting",
                    "word_count": len(content.split()),
                    "attempts": attempt + 1
                }
            ), None
            
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Direct prompting attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return Article(
                    title=topic,
                    content=f"# {topic}\n\nDirect prompting error after {max_retries} attempts: {error_msg}",
                    sections={},
                    metadata={"error": error_msg, "attempts": max_retries}
                ), error_msg

def main():
    parser = argparse.ArgumentParser(description="Enhanced STORM Baselines with Configuration")
    parser.add_argument("--method", choices=["direct", "storm", "all"], default="all")
    parser.add_argument("--num_topics", type=int, default=3)
    parser.add_argument("--provider", choices=["auto", "together", "groq", "huggingface", "openai"], 
                       default="auto", help="LLM provider preference")
    parser.add_argument("--config", default="storm_config.yaml", help="Configuration file path")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🌩️  Enhanced STORM Baselines with Configuration")
    
    # Initialize configuration manager
    config_manager = ConfigManager(args.config)
    
    # Create sample config if it doesn't exist
    if not Path(args.config).exists():
        logger.info(f"Creating sample configuration at {args.config}")
        config_manager.create_sample_config()
    
    logger.info(f"Provider preference: {args.provider}")
    logger.info(f"Methods: {args.method}")
    logger.info(f"Configuration: {args.config}")
    
    # Check environment setup
    available_providers = config_manager.get_available_providers()
    
    if not available_providers:
        logger.error("No API keys found in environment variables!")
        logger.error("Make sure your .env file contains at least one of:")
        for provider_name, provider_config in config_manager.providers.items():
            logger.error(f"  {provider_config.api_key_env}=your_api_key")
        return
    
    logger.info(f"Available providers: {', '.join(available_providers.keys())}")
    
    # Show current configuration
    best_provider = config_manager.get_best_provider(args.provider)
    if best_provider:
        logger.info(f"Selected provider: {best_provider.name} ({best_provider.model})")
        logger.info(f"STORM settings: conv_turns={config_manager.storm_config.max_conv_turn}, "
                   f"perspectives={config_manager.storm_config.max_perspective}, "
                   f"retries={config_manager.storm_config.max_retries}")
    
    # Create results directory
    results_dir = create_results_directory(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize STORM with configuration
    storm_runner = initialize_storm_with_retry(args, config_manager, max_retries=3)
    if not storm_runner:
        logger.error("Failed to initialize STORM with any provider")
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
            article, error = create_direct_baseline_with_retry(storm_runner, topic, config_manager)
            
            topic_results["generation_results"]["direct"] = {
                "success": error is None,
                "word_count": article.metadata.get("word_count", 0),
                "attempts": article.metadata.get("attempts", 1),
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
            article, error = run_storm_on_topic_with_retry(storm_runner, topic, storm_output_dir, config_manager)
            
            topic_results["generation_results"]["storm"] = {
                "success": error is None,
                "word_count": article.metadata.get("word_count", 0),
                "attempts": article.metadata.get("attempts", 1),
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
    
    # Save results with configuration info
    results_file = results_dir / "enhanced_storm_results.json"
    final_results = {
        "configuration": {
            "provider": best_provider.name if best_provider else "none",
            "model": best_provider.model if best_provider else "none", 
            "storm_settings": {
                "max_conv_turn": config_manager.storm_config.max_conv_turn,
                "max_perspective": config_manager.storm_config.max_perspective,
                "search_top_k": config_manager.storm_config.search_top_k,
                "enable_polish": config_manager.storm_config.enable_polish,
                "max_retries": config_manager.storm_config.max_retries
            }
        },
        "results": all_results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
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
        total_attempts = sum(r["generation_results"].get(method, {}).get("attempts", 0)
                           for r in all_results.values())
        total_words = sum(r["generation_results"].get(method, {}).get("word_count", 0)
                         for r in all_results.values()
                         if r["generation_results"].get(method, {}).get("success", False))
        avg_words = total_words / max(successes, 1)
        
        logger.info(f"{method.upper()}: {successes}/{len(all_results)} successful")
        logger.info(f"  Average words: {avg_words:.0f}")
        logger.info(f"  Total attempts: {total_attempts}")
        
        if evaluator and successes > 0:
            rouge_scores = []
            for result in all_results.values():
                if method in result["evaluation_results"]:
                    rouge_scores.append(result["evaluation_results"][method].get("rouge_1", 0))
            
            if rouge_scores:
                avg_rouge = sum(rouge_scores) / len(rouge_scores)
                logger.info(f"  Average ROUGE-1: {avg_rouge:.3f}")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"STORM outputs in: {storm_output_dir}")
    logger.info(f"Configuration used: {args.config}")

if __name__ == "__main__":
    main()