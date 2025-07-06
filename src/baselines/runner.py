"""
STORM runner with comprehensive SQLite avoidance for HPC environments.
"""
import os
import sys
import time
import logging
from pathlib import Path
from typing import List

def comprehensive_sqlite_avoidance():
    """Comprehensive SQLite avoidance for HPC environments."""
    
    # Step 1: Mock SQLite modules
    import types
    
    mock_sqlite3 = types.ModuleType('sqlite3')
    
    class MockConnection:
        def __init__(self, *args, **kwargs):
            pass
        def execute(self, *args, **kwargs):
            return self
        def executemany(self, *args, **kwargs):
            return self
        def fetchall(self):
            return []
        def fetchone(self):
            return None
        def commit(self):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    mock_sqlite3.connect = lambda *args, **kwargs: MockConnection()
    mock_sqlite3.Connection = MockConnection
    mock_sqlite3.Row = dict
    mock_sqlite3.PARSE_DECLTYPES = 1
    mock_sqlite3.PARSE_COLNAMES = 2
    
    sys.modules['sqlite3'] = mock_sqlite3
    sys.modules['_sqlite3'] = mock_sqlite3
    
    # Step 2: Disable LiteLLM caching BEFORE it's imported
    os.environ['LITELLM_DISABLE_CACHE'] = 'true'
    os.environ['LITELLM_CACHE_TYPE'] = 'none'
    
    # Step 3: Mock diskcache to avoid SQLite usage
    mock_diskcache = types.ModuleType('diskcache')
    
    class MockDiskCache:
        def __init__(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            return None
        def set(self, *args, **kwargs):
            pass
        def delete(self, *args, **kwargs):
            pass
        def clear(self, *args, **kwargs):
            pass
        def close(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __getitem__(self, key):
            raise KeyError(key)
        def __setitem__(self, key, value):
            pass
        def __delitem__(self, key):
            pass
        def __contains__(self, key):
            return False
    
    mock_diskcache.Cache = MockDiskCache
    mock_diskcache.FanoutCache = MockDiskCache
    sys.modules['diskcache'] = mock_diskcache
    sys.modules['diskcache.core'] = mock_diskcache
    
    # Step 4: Set up file system for cache-free operation
    work_dir = os.getcwd()
    
    # Override HOME to avoid permission issues
    os.environ['HOME'] = work_dir
    os.environ['TMPDIR'] = os.path.join(work_dir, 'tmp')
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    
    # Disable various caching mechanisms
    cache_disable_vars = [
        'DSP_CACHEDIR', 'JOBLIB_CACHE_DIR', 'DSPY_CACHEDIR',
        'DSP_CACHE_DISABLED', 'DSPY_DISABLE_CACHE', 'DSPY_DISABLE_OPTIMIZERS',
        'DSPY_DISABLE_TELEMETRY', 'LITELLM_LOG', 'LITELLM_CACHE'
    ]
    
    for var in cache_disable_vars:
        if var in ['LITELLM_LOG']:
            os.environ[var] = 'INFO'  # Set valid log level
        else:
            os.environ[var] = ''
    
    os.environ['DSP_CACHE_DISABLED'] = '1'
    os.environ['DSPY_DISABLE_CACHE'] = '1'
    
    print("✅ Comprehensive SQLite avoidance configured")

# Apply SQLite avoidance BEFORE any imports
comprehensive_sqlite_avoidance()

from handlers import QwenQueryHandler
from utils.data_models import Article
from baselines.mock_search import MockSearchRM
from baselines.llm_wrapper import UnifiedLocalLLMWrapper

logger = logging.getLogger(__name__)


class BaselinesRunner:
    """
    STORM runner with comprehensive SQLite avoidance.
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
        
        logger.info("✅ Workspace configured")
    
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
        """Run STORM with comprehensive SQLite avoidance."""
        logger.info(f"⛈️  Running STORM for: {topic}")
        
        try:
            # Step 1: Additional runtime cache disabling
            logger.info("🚫 Disabling runtime caching...")
            
            # Disable LiteLLM caching at runtime
            try:
                import litellm
                litellm.cache = None
                logger.info("✅ LiteLLM cache disabled")
            except:
                logger.info("⚠️ LiteLLM not yet imported")
            
            # Step 2: Create LiteLLM wrapper
            litellm_wrapper = UnifiedLocalLLMWrapper(
                self.query_handler, 
                max_tokens=self.config.max_new_tokens, 
                temperature=self.config.temperature
            )
            
            logger.info("✅ Created LiteLLM wrapper")
            
            # Step 3: Import STORM components (after all mocking)
            logger.info("📦 Importing STORM components...")
            from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
            logger.info("✅ STORM components imported successfully")
            
            # Step 4: Configure STORM
            lm_config = STORMWikiLMConfigs()
            
            # Use the wrapper for all STORM components
            lm_config.set_conv_simulator_lm(litellm_wrapper)
            lm_config.set_question_asker_lm(litellm_wrapper)
            lm_config.set_outline_gen_lm(litellm_wrapper)
            lm_config.set_article_gen_lm(litellm_wrapper)
            lm_config.set_article_polish_lm(litellm_wrapper)
            
            logger.info("✅ STORM LM config set up")
            
            # Step 5: Setup search
            search_rm = MockSearchRM(k=self.config.search_top_k)
            
            # Step 6: Create STORM runner
            engine_args = STORMWikiRunnerArguments(
                output_dir=self.storm_output_dir,
                max_conv_turn=self.config.max_conv_turn,
                max_perspective=self.config.max_perspective,
                search_top_k=self.config.search_top_k,
                max_thread_num=self.config.max_thread_num
            )
            
            storm_runner = STORMWikiRunner(engine_args, lm_config, search_rm)
            
            # Step 7: Run STORM
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
            
            # Step 8: Process output
            return self._process_storm_output(topic, generation_time)
            
        except Exception as e:
            logger.error(f"❌ STORM failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
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