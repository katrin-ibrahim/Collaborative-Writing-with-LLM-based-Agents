"""
Simplified baselines runner that works with STORM's DSPy version.

This replaces the complex wrapper architecture with a simple approach
that uses the correct DSPy version STORM expects.
"""
import os
import time
import logging
from pathlib import Path
from typing import List

from handlers import QwenQueryHandler
from utils.data_models import Article
from baselines.mock_search import MockSearchRM
from baselines.dspy_integration import setup_dspy_for_storm

logger = logging.getLogger(__name__)


class BaselinesRunner:
    """
    Simplified runner that works with STORM's required DSPy version.
    
    This approach is much cleaner than trying to adapt to newer DSPy APIs.
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
        
        # Setup workspace
        self._setup_workspace()
    
    def _setup_workspace(self):
        """Setup working directories."""
        self.work_dir = os.getcwd()
        self.storm_output_dir = os.path.join(self.work_dir, "storm_output")
        os.makedirs(self.storm_output_dir, exist_ok=True)
        
        # Environment setup for STORM
        self.old_env = {
            "HOME": os.environ.get("HOME", ""),
            "TMPDIR": os.environ.get("TMPDIR", "")
        }
        
        os.environ["HOME"] = self.work_dir
        os.environ["TMPDIR"] = os.path.join(self.work_dir, "tmp")
        os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    
    def _restore_environment(self):
        """Restore original environment variables."""
        for key, value in self.old_env.items():
            if value:
                os.environ[key] = value
    
    def run_direct_prompting(self, topic: str) -> Article:
        """Run direct prompting baseline."""
        logger.info(f"🔤 Running Direct Prompting for: {topic}")
        
        prompt = f"""Write a comprehensive, well-structured article about "{topic}".

Requirements:
1. Create a detailed article with multiple sections
2. Use only your internal knowledge
3. Include introduction, main sections, and conclusion
4. Write in encyclopedic style similar to Wikipedia
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
            
            # Post-processing
            if content and not content.startswith("#"):
                content = f"# {topic}\n\n{content}"
            elif not content:
                content = f"# {topic}\n\nError: No content generated"
            
            return Article(
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
            
        except Exception as e:
            logger.error(f"❌ Direct Prompting failed: {e}")
            return Article(
                title=topic,
                content=f"# {topic}\n\nError: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "direct_prompting"}
            )
    
    def run_storm(self, topic: str) -> Article:
        """Run STORM using the correct DSPy version (2.4.9)."""
        logger.info(f"⛈️  Running STORM for: {topic}")
        
        try:
            # Step 1: Setup DSPy with correct version
            storm_lm = setup_dspy_for_storm(
                self.query_handler, 
                self.config.max_new_tokens, 
                self.config.temperature
            )
            
            if not storm_lm:
                raise RuntimeError("Failed to setup DSPy for STORM")
            
            # Step 2: Import STORM components
            from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
            
            # Step 3: Configure STORM with the DSPy LM
            lm_config = STORMWikiLMConfigs()
            lm_config.set_conv_simulator_lm(storm_lm)
            lm_config.set_question_asker_lm(storm_lm)
            lm_config.set_outline_gen_lm(storm_lm)
            lm_config.set_article_gen_lm(storm_lm)
            lm_config.set_article_polish_lm(storm_lm)
            
            # Step 4: Setup search
            search_rm = MockSearchRM(k=self.config.search_top_k)
            
            # Step 5: Create STORM runner
            engine_args = STORMWikiRunnerArguments(
                output_dir=self.storm_output_dir,
                max_conv_turn=self.config.max_conv_turn,
                max_perspective=self.config.max_perspective,
                search_top_k=self.config.search_top_k,
                max_thread_num=self.config.max_thread_num
            )
            
            storm_runner = STORMWikiRunner(engine_args, lm_config, search_rm)
            
            # Step 6: Run STORM
            logger.info(f"🚀 Executing STORM for: {topic}")
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
            
            # Step 7: Process output
            return self._process_storm_output(topic, generation_time)
            
        except Exception as e:
            logger.error(f"❌ STORM failed: {e}")
            return Article(
                title=topic,
                content=f"# {topic}\n\nSTORM Error: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "storm_local"}
            )
    
    def _process_storm_output(self, topic: str, generation_time: float) -> Article:
        """Process STORM output files into Article object."""
        topic_subdir = Path(self.storm_output_dir) / topic.replace(" ", "_").replace("/", "_")
        
        # Find generated content
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
        
        return Article(
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
    
    def run_all_baselines(self, topics: List[str], methods: List[str] = None) -> dict:
        """Run all specified baseline methods on topics."""
        if methods is None:
            methods = ["direct_prompting", "storm"]
        
        logger.info(f"🚀 Running {len(methods)} methods on {len(topics)} topics")
        
        all_results = {}
        
        try:
            for i, topic in enumerate(topics, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"📝 Processing {i}/{len(topics)}: {topic}")
                logger.info(f"{'='*60}")
                
                topic_results = {}
                
                for method in methods:
                    logger.info(f"▶️  Running {method}...")
                    
                    try:
                        article = self._run_method(method, topic)
                        
                        topic_results[method] = {
                            "article": article,
                            "word_count": article.metadata.get("word_count", 0),
                            "success": "error" not in article.metadata
                        }
                        
                        success = topic_results[method]["success"]
                        word_count = topic_results[method]["word_count"]
                        
                        if success:
                            logger.info(f"✅ {method} completed successfully ({word_count} words)")
                        else:
                            logger.warning(f"⚠️  {method} completed with errors")
                    
                    except Exception as e:
                        logger.error(f"❌ {method} failed: {e}")
                        topic_results[method] = self._create_error_result(topic, method, str(e))
                
                all_results[topic] = topic_results
                
                # Delay between topics
                if i < len(topics):
                    delay = 10.0
                    logger.info(f"⏳ Waiting {delay}s before next topic...")
                    time.sleep(delay)
            
            self._log_summary(all_results, methods)
            return all_results
            
        finally:
            self._restore_environment()
    
    def _run_method(self, method: str, topic: str) -> Article:
        """Run specific baseline method."""
        if method == "direct_prompting":
            return self.run_direct_prompting(topic)
        elif method == "storm":
            return self.run_storm(topic)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _create_error_result(self, topic: str, method: str, error: str) -> dict:
        """Create error result structure."""
        return {
            "article": Article(
                title=topic,
                content=f"# {topic}\n\n{method} failed: {error}",
                sections={},
                metadata={"error": error, "method": method}
            ),
            "word_count": 0,
            "success": False
        }
    
    def _log_summary(self, all_results: dict, methods: List[str]):
        """Log execution summary."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 EXECUTION SUMMARY")
        logger.info(f"{'='*60}")
        
        for method in methods:
            successes = sum(1 for r in all_results.values() 
                           if r.get(method, {}).get("success", False))
            total_words = sum(r.get(method, {}).get("word_count", 0)
                             for r in all_results.values()
                             if r.get(method, {}).get("success", False))
            avg_words = total_words / max(successes, 1)
            
            logger.info(f"📈 {method}: {successes}/{len(all_results)} successful, {avg_words:.0f} avg words")