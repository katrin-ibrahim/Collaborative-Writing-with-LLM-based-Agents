import argparse
import logging
import json
import os
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

from utils.freshwiki_loader import FreshWikiLoader
from evaluation.evaluator import ArticleEvaluator
from utils.logging_setup import setup_logging
from utils.data_models import Article
from config.storm_config import load_config
from handlers import QwenQueryHandler


class LocalLiteLLMWrapper:
    """Wrapper to make QwenQueryHandler compatible with STORM's LiteLLM interface."""
    
    def __init__(self, query_handler: QwenQueryHandler, max_tokens: int = 512, temperature: float = 0.7):
        self.query_handler = query_handler
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def complete(self, messages, max_tokens=None, temperature=None, **kwargs):
        """LiteLLM-compatible completion method."""
        # Extract prompt from messages
        prompt_parts = []
        system_prompt = None
        
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content")
            elif message.get("role") == "user":
                prompt_parts.append(message.get("content"))
        
        prompt = "\n".join(prompt_parts)
        
        # Use provided parameters or defaults
        actual_max_tokens = max_tokens or self.max_tokens
        actual_temperature = temperature or self.temperature
        
        # Call our query handler
        response = self.query_handler.query(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=actual_max_tokens,
            temperature=actual_temperature
        )
        
        # Return in LiteLLM format
        return type('Response', (), {
            'choices': [type('Choice', (), {
                'message': type('Message', (), {
                    'content': response
                })()
            })()]
        })()


class BaselinesRunner:
    """Runner for all baseline methods."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize query handler
        if config.model_type == "local":
            self.query_handler = QwenQueryHandler(config.local_model_path)
            if not self.query_handler.is_available():
                raise RuntimeError("Local Qwen model not available")
            self.logger.info("Using local Qwen model")
        else:
            raise NotImplementedError("Only local models supported in this version")
    
    def run_direct_prompting(self, topic: str) -> Article:
        """Run direct prompting baseline - pure internal knowledge."""
        self.logger.info(f"Running Direct Prompting for: {topic}")
        
        prompt = f"""Write a comprehensive, well-structured article about "{topic}".

Requirements:
1. Create a detailed article with multiple sections
2. Use only your internal knowledge (no external sources needed)
3. Include an introduction, several main sections, and a conclusion
4. Write in an encyclopedic style similar to Wikipedia
5. Aim for 800-1200 words
6. Use clear headings and subheadings

Topic: {topic}

Article:"""

        try:
            start_time = time.time()
            content = self.query_handler.query(
                prompt=prompt,
                max_new_tokens=1024,
                temperature=0.7
            )
            generation_time = time.time() - start_time
            
            # Basic post-processing
            if not content.startswith("#"):
                content = f"# {topic}\n\n{content}"
            
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct_prompting",
                    "word_count": len(content.split()),
                    "generation_time": generation_time,
                    "model": "qwen_local"
                }
            )
            
            self.logger.info(f"Direct Prompting completed: {len(content.split())} words in {generation_time:.1f}s")
            return article
            
        except Exception as e:
            self.logger.error(f"Direct Prompting failed: {e}")
            return Article(
                title=topic,
                content=f"# {topic}\n\nError in direct prompting: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "direct_prompting"}
            )
    
    def run_storm(self, topic: str) -> Article:
        """Run STORM baseline using local model."""
        self.logger.info(f"Running STORM for: {topic}")
        
        # Fix permission issues - set writable directories
        work_dir = os.getcwd()
        storm_output_dir = os.path.join(work_dir, "storm_output")
        os.makedirs(storm_output_dir, exist_ok=True)
        
        # Override environment variables that might cause permission issues
        old_home = os.environ.get("HOME", "")
        old_tmpdir = os.environ.get("TMPDIR", "")
        
        os.environ["HOME"] = work_dir
        os.environ["TMPDIR"] = os.path.join(work_dir, "tmp")
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
        
        self.logger.info(f"Set HOME from {old_home} to {os.environ['HOME']}")
        self.logger.info(f"Set TMPDIR from {old_tmpdir} to {os.environ['TMPDIR']}")
        self.logger.info(f"Storm output dir: {storm_output_dir}")
        
        try:
            import sys
            try:
                import pysqlite3 as sqlite3
                sys.modules['sqlite3'] = sqlite3
                self.logger.info("✅ Using pysqlite3 as sqlite3 replacement for STORM")
            except ImportError:
                self.logger.warning("⚠️ pysqlite3 not available, STORM may fail")
            
            # Import STORM components
            from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
            from knowledge_storm.rm import DuckDuckGoSearchRM
            
            # Create LiteLLM wrapper
            llm_wrapper = LocalLiteLLMWrapper(
                self.query_handler,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature
            )
            
            # Set up STORM LM configs with our wrapper
            lm_config = STORMWikiLMConfigs()
            lm_config.set_conv_simulator_lm(llm_wrapper)
            lm_config.set_question_asker_lm(llm_wrapper)
            lm_config.set_outline_gen_lm(llm_wrapper)
            lm_config.set_article_gen_lm(llm_wrapper)
            lm_config.set_article_polish_lm(llm_wrapper)
            
            # Set up retrieval
            rm = DuckDuckGoSearchRM(k=self.config.search_top_k)
            
            # Create STORM runner
            engine_args = STORMWikiRunnerArguments(
                output_dir=storm_output_dir,
                max_conv_turn=self.config.max_conv_turn,
                max_perspective=self.config.max_perspective,
                search_top_k=self.config.search_top_k,
                max_thread_num=self.config.max_thread_num
            )
            
            storm_runner = STORMWikiRunner(engine_args, lm_config, rm)
            
            # Run STORM
            start_time = time.time()
            storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=self.config.enable_polish
            )
            generation_time = time.time() - start_time
            
            # Read generated article
            topic_subdir = Path(storm_output_dir) / topic.replace(" ", "_").replace("/", "_")
            
            article_files = [
                topic_subdir / "storm_gen_article_polished.txt",
                topic_subdir / "storm_gen_article.txt"
            ]
            
            content = None
            for article_file in article_files:
                if article_file.exists():
                    content = article_file.read_text(encoding='utf-8')
                    break
            
            if not content:
                content = f"# {topic}\n\nSTORM completed but no article content found"
            
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm_local",
                    "word_count": len(content.split()),
                    "generation_time": generation_time,
                    "model": "qwen_local",
                    "output_dir": str(topic_subdir)
                }
            )
            
            self.logger.info(f"STORM completed: {len(content.split())} words in {generation_time:.1f}s")
            return article
            
        except Exception as e:
            self.logger.error(f"STORM failed: {e}")
            self.logger.error(f"Exception type: {type(e)}")
            
            # If it's a permission error, try to get more details
            if "Permission denied" in str(e):
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Restore environment
            if old_home:
                os.environ["HOME"] = old_home
            if old_tmpdir:
                os.environ["TMPDIR"] = old_tmpdir
                
            return Article(
                title=topic,
                content=f"# {topic}\n\nSTORM Error: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "storm_local"}
            )
    
    def run_all_baselines(self, topics, methods=None):
        """Run specified baselines on all topics."""
        if methods is None:
            methods = ["direct_prompting", "storm"]
        
        all_results = {}
        
        for i, topic in enumerate(topics, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing {i}/{len(topics)}: {topic}")
            self.logger.info(f"{'='*60}")
            
            topic_results = {}
            
            # Run each baseline
            for method in methods:
                self.logger.info(f"Running {method}...")
                
                if method == "direct_prompting":
                    article = self.run_direct_prompting(topic)
                elif method == "storm":
                    article = self.run_storm(topic)
                else:
                    self.logger.warning(f"Unknown method: {method}")
                    continue
                
                topic_results[method] = {
                    "article": article,
                    "word_count": article.metadata.get("word_count", 0),
                    "success": "error" not in article.metadata
                }
            
            all_results[topic] = topic_results
            
            # Add delay between topics
            if i < len(topics):
                delay = 10.0  # Shorter delay for local models
                self.logger.info(f"Waiting {delay}s before next topic...")
                time.sleep(delay)
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Baselines Runner with Local Models")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--num_topics", type=int, default=5)
    parser.add_argument("--methods", nargs="+", default=["direct_prompting", "storm"], 
                        help="Baselines to run")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("🚀 Baselines Runner with Local Models")
    logger.info(f"Methods: {', '.join(args.methods)}")

    # Load config
    config = load_config(args.config)
    logger.info(f"Model type: {config.model_type}")
    if config.model_type == "local":
        logger.info(f"Local model path: {config.local_model_path}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    methods_str = "_".join(args.methods)
    results_dir = Path("results") / f"{methods_str}_{args.num_topics}topics_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load topics
    logger.info("Loading topics from FreshWiki...")
    freshwiki_loader = FreshWikiLoader()
    entries = freshwiki_loader.get_evaluation_sample(args.num_topics)

    if not entries:
        logger.error("No FreshWiki entries found")
        return

    logger.info(f"Loaded {len(entries)} topics")

    # Initialize runner
    try:
        runner = BaselinesRunner(config)
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        return

    # Initialize evaluator
    evaluator = None
    if not args.skip_evaluation:
        evaluator = ArticleEvaluator()
        logger.info("Evaluation enabled")

    # Run baselines
    topics = [entry.topic for entry in entries]
    all_results = runner.run_all_baselines(topics, args.methods)

    # Evaluate results
    final_results = {}
    for topic, baseline_results in all_results.items():
        topic_result = {"baselines": {}}
        
        # Get corresponding entry for evaluation
        entry = next((e for e in entries if e.topic == topic), None)
        
        for method, result in baseline_results.items():
            method_result = {
                "generation_results": {
                    "success": result["success"],
                    "word_count": result["word_count"],
                    "metadata": result["article"].metadata
                },
                "evaluation_results": {}
            }
            
            # Evaluate if possible
            if evaluator and result["success"] and entry:
                try:
                    metrics = evaluator.evaluate_article(result["article"], entry)
                    method_result["evaluation_results"] = metrics
                except Exception as e:
                    logger.warning(f"Evaluation failed for {method} on {topic}: {e}")
            
            topic_result["baselines"][method] = method_result
        
        final_results[topic] = topic_result

    # Save results
    results_file = results_dir / "results.json"
    output_data = {
        "configuration": {
            "model_type": config.model_type,
            "local_model_path": config.local_model_path if config.model_type == "local" else None,
            "methods": args.methods,
            "storm_settings": {
                "max_conv_turn": config.max_conv_turn,
                "max_perspective": config.max_perspective,
                "search_top_k": config.search_top_k,
                "enable_polish": config.enable_polish
            }
        },
        "results": final_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")

    for method in args.methods:
        successes = sum(1 for r in final_results.values() 
                       if r["baselines"].get(method, {}).get("generation_results", {}).get("success", False))
        total_words = sum(r["baselines"].get(method, {}).get("generation_results", {}).get("word_count", 0)
                         for r in final_results.values()
                         if r["baselines"].get(method, {}).get("generation_results", {}).get("success", False))
        avg_words = total_words / max(successes, 1)
        
        logger.info(f"{method}: {successes}/{len(final_results)} successful, {avg_words:.0f} avg words")

    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()