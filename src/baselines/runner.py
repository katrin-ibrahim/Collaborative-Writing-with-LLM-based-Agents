"""
Clean baselines runner for LLM evaluation.

This module provides a clean implementation of the BaselinesRunner
that can run various baseline methods like direct prompting and STORM.
"""

import os
import time
import logging
from pathlib import Path
from typing import List

from handlers import QwenQueryHandler
from utils.data_models import Article
from .mock_search import MockSearchRM
from .llm_wrapper import LocalLiteLLMWrapper
from .dspy_integration import setup_dspy_integration

logger = logging.getLogger(__name__)


class BaselinesRunner:
    """
    Clean runner for baseline evaluation methods.
    
    Supports:
    - Direct prompting (internal knowledge only)
    - STORM (with local model integration)
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize query handler
        if config.model_type == "local":
            self.query_handler = QwenQueryHandler(config.local_model_path)
            if not self.query_handler.is_available():
                raise RuntimeError("Local Qwen model not available")
            logger.info("✅ Local Qwen model loaded successfully")
        else:
            raise NotImplementedError("Only local models supported in this version")
    
    def run_direct_prompting(self, topic: str) -> Article:
        """
        Run direct prompting baseline using only internal knowledge.
        
        Args:
            topic: Topic to write about
            
        Returns:
            Generated Article object
        """
        logger.info(f"🔤 Running Direct Prompting for: {topic}")
        
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
            if content and not content.startswith("#"):
                content = f"# {topic}\n\n{content}"
            elif not content:
                content = f"# {topic}\n\nError: No content generated"
            
            word_count = len(content.split())
            
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct_prompting",
                    "word_count": word_count,
                    "generation_time": generation_time,
                    "model": "qwen_local"
                }
            )
            
            logger.info(f"✅ Direct Prompting completed: {word_count} words in {generation_time:.1f}s")
            return article
            
        except Exception as e:
            logger.error(f"❌ Direct Prompting failed: {e}")
            return Article(
                title=topic,
                content=f"# {topic}\n\nError in direct prompting: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "direct_prompting"}
            )
    
    def run_storm(self, topic: str) -> Article:
        """
        Run STORM baseline using local model with full integration.
        
        Args:
            topic: Topic to write about
            
        Returns:
            Generated Article object
        """
        logger.info(f"⛈️  Running STORM for: {topic}")
        
        # Setup working directories
        work_dir = os.getcwd()
        storm_output_dir = os.path.join(work_dir, "storm_output")
        os.makedirs(storm_output_dir, exist_ok=True)
        
        # Set environment variables to avoid permission issues
        old_home = os.environ.get("HOME", "")
        old_tmpdir = os.environ.get("TMPDIR", "")
        
        os.environ["HOME"] = work_dir
        os.environ["TMPDIR"] = os.path.join(work_dir, "tmp")
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
        
        logger.debug(f"Storm output dir: {storm_output_dir}")
        logger.debug(f"Set HOME: {os.environ['HOME']}")
        logger.debug(f"Set TMPDIR: {os.environ['TMPDIR']}")
        
        try:
            # Step 1: Setup DSPy integration
            logger.info("🔧 Setting up DSPy integration...")
            if not setup_dspy_integration(
                self.query_handler, 
                self.config.max_new_tokens, 
                self.config.temperature
            ):
                raise RuntimeError("Failed to setup DSPy integration")
            
            # Step 2: Import STORM components (after DSPy setup)
            logger.info("📦 Importing STORM components...")
            from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
            
            # Step 3: Create LLM wrapper
            logger.info("🔗 Creating LLM wrapper...")
            llm_wrapper = LocalLiteLLMWrapper(
                self.query_handler,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature
            )
            
            # Step 4: Configure STORM LM settings
            logger.info("⚙️  Configuring STORM LM settings...")
            lm_config = STORMWikiLMConfigs()
            lm_config.set_conv_simulator_lm(llm_wrapper)
            lm_config.set_question_asker_lm(llm_wrapper)
            lm_config.set_outline_gen_lm(llm_wrapper)
            lm_config.set_article_gen_lm(llm_wrapper)
            lm_config.set_article_polish_lm(llm_wrapper)
            
            # Step 5: Setup mock search (to avoid DuckDuckGo rate limiting)
            logger.info("🔍 Setting up mock search...")
            rm = MockSearchRM(k=self.config.search_top_k)
            
            # Step 6: Create STORM runner
            logger.info("🏃 Creating STORM runner...")
            engine_args = STORMWikiRunnerArguments(
                output_dir=storm_output_dir,
                max_conv_turn=self.config.max_conv_turn,
                max_perspective=self.config.max_perspective,
                search_top_k=self.config.search_top_k,
                max_thread_num=self.config.max_thread_num
            )
            
            storm_runner = STORMWikiRunner(engine_args, lm_config, rm)
            
            # Step 7: Run STORM
            logger.info(f"🚀 Starting STORM run for topic: {topic}")
            start_time = time.time()
            
            result = storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=self.config.enable_polish
            )
            
            generation_time = time.time() - start_time
            logger.info(f"⏱️  STORM completed in {generation_time:.1f}s")
            
            # Step 8: Read generated article
            article = self._read_storm_output(topic, storm_output_dir, generation_time)
            
            return article
            
        except Exception as e:
            logger.error(f"❌ STORM failed: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            return Article(
                title=topic,
                content=f"# {topic}\n\nSTORM Error: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "storm_local"}
            )
            
        finally:
            # Restore environment variables
            if old_home:
                os.environ["HOME"] = old_home
            if old_tmpdir:
                os.environ["TMPDIR"] = old_tmpdir
    
    def _read_storm_output(self, topic: str, storm_output_dir: str, generation_time: float) -> Article:
        """
        Read STORM output files and create Article object.
        
        Args:
            topic: The topic that was processed
            storm_output_dir: Directory containing STORM output
            generation_time: Time taken for generation
            
        Returns:
            Article object with STORM results
        """
        topic_subdir = Path(storm_output_dir) / topic.replace(" ", "_").replace("/", "_")
        
        # Try to find generated article files in order of preference
        article_files = [
            topic_subdir / "storm_gen_article_polished.txt",
            topic_subdir / "storm_gen_article.txt"
        ]
        
        content = None
        for article_file in article_files:
            if article_file.exists():
                try:
                    content = article_file.read_text(encoding='utf-8')
                    logger.info(f"📄 Read STORM output from: {article_file.name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to read {article_file}: {e}")
        
        if not content:
            logger.warning("No STORM article content found")
            content = f"# {topic}\n\nSTORM completed but no article content found"
        
        word_count = len(content.split())
        
        return Article(
            title=topic,
            content=content,
            sections={},
            metadata={
                "method": "storm_local",
                "word_count": word_count,
                "generation_time": generation_time,
                "model": "qwen_local",
                "output_dir": str(topic_subdir)
            }
        )
    
    def run_all_baselines(self, topics: List[str], methods: List[str] = None) -> dict:
        """
        Run specified baseline methods on all topics.
        
        Args:
            topics: List of topics to process
            methods: List of methods to run (default: ["direct_prompting", "storm"])
            
        Returns:
            Dictionary with results for each topic and method
        """
        if methods is None:
            methods = ["direct_prompting", "storm"]
        
        logger.info(f"🚀 Running {len(methods)} methods on {len(topics)} topics")
        
        all_results = {}
        
        for i, topic in enumerate(topics, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📝 Processing {i}/{len(topics)}: {topic}")
            logger.info(f"{'='*60}")
            
            topic_results = {}
            
            # Run each baseline method
            for method in methods:
                logger.info(f"▶️  Running {method}...")
                
                try:
                    if method == "direct_prompting":
                        article = self.run_direct_prompting(topic)
                    elif method == "storm":
                        article = self.run_storm(topic)
                    else:
                        logger.warning(f"⚠️  Unknown method: {method}")
                        continue
                    
                    topic_results[method] = {
                        "article": article,
                        "word_count": article.metadata.get("word_count", 0),
                        "success": "error" not in article.metadata
                    }
                    
                    if topic_results[method]["success"]:
                        logger.info(f"✅ {method} completed successfully ({topic_results[method]['word_count']} words)")
                    else:
                        logger.warning(f"⚠️  {method} completed with errors")
                        
                except Exception as e:
                    logger.error(f"❌ {method} failed with exception: {e}")
                    topic_results[method] = {
                        "article": Article(
                            title=topic,
                            content=f"# {topic}\n\n{method} failed: {str(e)}",
                            sections={},
                            metadata={"error": str(e), "method": method}
                        ),
                        "word_count": 0,
                        "success": False
                    }
            
            all_results[topic] = topic_results
            
            # Add delay between topics to avoid overwhelming the system
            if i < len(topics):
                delay = 10.0
                logger.info(f"⏳ Waiting {delay}s before next topic...")
                time.sleep(delay)
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 SUMMARY")
        logger.info(f"{'='*60}")
        
        for method in methods:
            successes = sum(1 for r in all_results.values() 
                           if r.get(method, {}).get("success", False))
            total_words = sum(r.get(method, {}).get("word_count", 0)
                             for r in all_results.values()
                             if r.get(method, {}).get("success", False))
            avg_words = total_words / max(successes, 1)
            
            logger.info(f"📈 {method}: {successes}/{len(all_results)} successful, {avg_words:.0f} avg words")
        
        return all_results