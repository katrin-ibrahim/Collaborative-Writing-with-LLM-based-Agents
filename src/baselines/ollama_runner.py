import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils.ollama_client import OllamaClient
from config.model_config import ModelConfig
from utils.data_models import Article
from baselines.mock_search import MockSearchRM

logger = logging.getLogger(__name__)


class OllamaBaselinesRunner:
    """Simplified runner for Ollama-based baseline experiments."""
    
    def __init__(self, ollama_host: str = "http://10.167.31.201:11434/", 
                 model_config: Optional[ModelConfig] = None):
        self.ollama_host = ollama_host
        self.model_config = model_config or ModelConfig()
        
        # Initialize Ollama client
        self.client = OllamaClient(host=ollama_host)
        if not self.client.is_available():
            raise RuntimeError(f"Ollama server not available at {ollama_host}")
        
        # Log available models
        available_models = self.client.list_models()
        logger.info(f"Connected to Ollama with {len(available_models)} models available")
        
        # Setup workspace
        self.work_dir = os.getcwd()
        self.output_dir = os.path.join(self.work_dir, "ollama_output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_direct_prompting(self, topic: str) -> Article:
        """Run direct prompting baseline with Ollama."""
        logger.info(f"Running Direct Prompting for: {topic}")
        
        prompt = f"""Write a comprehensive, well-structured article about "{topic}".

Requirements:
1. Create a detailed article with multiple sections
2. Include an introduction, at least 4-5 main sections, and a conclusion
3. Each section should have 2-3 paragraphs
4. Use clear markdown formatting with # for title and ## for sections
5. Write in an encyclopedic, informative style
6. Aim for 1000-1500 words total
7. Include specific details, examples, and explanations

Format:
# {topic}

[Introduction paragraph]

## [Section 1 Title]
[Content...]

## [Section 2 Title]
[Content...]

[Continue with more sections...]

## Conclusion
[Concluding paragraph]

Now write the article:"""

        try:
            start_time = time.time()
            
            # Use writing model for direct prompting
            model = self.model_config.get_model_for_task("writing")
            temperature = self.model_config.get_temperature_for_task("writing")
            
            content = self.client.call_api(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=2048
            )
            
            generation_time = time.time() - start_time
            
            # Ensure proper formatting
            if content and not content.startswith("#"):
                content = f"# {topic}\n\n{content}"
            
            return Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct_prompting",
                    "model": model,
                    "word_count": len(content.split()),
                    "generation_time": generation_time,
                    "temperature": temperature
                }
            )
            
        except Exception as e:
            logger.error(f"Direct Prompting failed: {e}")
            return Article(
                title=topic,
                content=f"# {topic}\n\nError in generation: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "direct_prompting"}
            )
    
    def run_storm(self, topic: str) -> Article:
        """Run STORM with Ollama models."""
        logger.info(f"Running STORM for: {topic}")
        
        try:
            # Import STORM components
            from knowledge_storm import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
            from baselines.llm_wrapper import OllamaLiteLLMWrapper
            
            # Configure STORM with task-specific models
            lm_config = STORMWikiLMConfigs()
            
            # Create wrappers for different tasks
            research_wrapper = OllamaLiteLLMWrapper(
                self.client, 
                model=self.model_config.get_model_for_task("research"),
                temperature=self.model_config.get_temperature_for_task("research")
            )
            
            outline_wrapper = OllamaLiteLLMWrapper(
                self.client,
                model=self.model_config.get_model_for_task("outline"),
                temperature=self.model_config.get_temperature_for_task("outline")
            )
            
            writing_wrapper = OllamaLiteLLMWrapper(
                self.client,
                model=self.model_config.get_model_for_task("writing"),
                temperature=self.model_config.get_temperature_for_task("writing")
            )
            
            polish_wrapper = OllamaLiteLLMWrapper(
                self.client,
                model=self.model_config.get_model_for_task("polish"),
                temperature=self.model_config.get_temperature_for_task("polish")
            )
            
            # Set different models for different STORM components
            lm_config.set_conv_simulator_lm(research_wrapper)
            lm_config.set_question_asker_lm(research_wrapper)
            lm_config.set_outline_gen_lm(outline_wrapper)
            lm_config.set_article_gen_lm(writing_wrapper)
            lm_config.set_article_polish_lm(polish_wrapper)
            
            # Setup search (using mock for now)
            search_rm = MockSearchRM(k=3)
            
            # Configure STORM runner
            storm_output_dir = os.path.join(self.output_dir, "storm")
            os.makedirs(storm_output_dir, exist_ok=True)
            
            engine_args = STORMWikiRunnerArguments(
                output_dir=storm_output_dir,
                max_conv_turn=3,
                max_perspective=3,
                search_top_k=3,
                max_thread_num=1
            )
            
            storm_runner = STORMWikiRunner(engine_args, lm_config, search_rm)
            
            # Run STORM
            start_time = time.time()
            
            storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True
            )
            
            generation_time = time.time() - start_time
            
            # Extract generated content
            content = self._extract_storm_output(topic, storm_output_dir)
            
            return Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "models": {
                        "research": self.model_config.research_model,
                        "outline": self.model_config.outline_model,
                        "writing": self.model_config.writing_model,
                        "polish": self.model_config.polish_model
                    },
                    "word_count": len(content.split()),
                    "generation_time": generation_time
                }
            )
            
        except Exception as e:
            logger.error(f"STORM failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return Article(
                title=topic,
                content=f"# {topic}\n\nSTORM Error: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "storm"}
            )
    
    def run_self_rag(self, topic: str) -> Article:
        """Run Self-RAG with Ollama."""
        logger.info(f"Running Self-RAG for: {topic}")
        
        try:
            from baselines.self_rag_runner import SelfRAGRunner
            
            # Initialize Self-RAG with Ollama
            self_rag = SelfRAGRunner(
                ollama_client=self.client,
                model_config=self.model_config
            )
            
            # Run Self-RAG
            start_time = time.time()
            content = self_rag.generate_with_self_rag(topic)
            generation_time = time.time() - start_time
            
            return Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "self_rag",
                    "models": {
                        "retrieval": self.model_config.retrieval_model,
                        "generation": self.model_config.generation_model,
                        "reflection": self.model_config.reflection_model
                    },
                    "word_count": len(content.split()),
                    "generation_time": generation_time
                }
            )
            
        except Exception as e:
            logger.error(f"Self-RAG failed: {e}")
            return Article(
                title=topic,
                content=f"# {topic}\n\nSelf-RAG Error: {str(e)}",
                sections={},
                metadata={"error": str(e), "method": "self_rag"}
            )
    
    def _extract_storm_output(self, topic: str, storm_output_dir: str) -> str:
        """Extract generated content from STORM output."""
        topic_dir = Path(storm_output_dir) / topic.replace(" ", "_").replace("/", "_")
        
        # Try different output files in order of preference
        output_files = [
            "storm_gen_article_polished.txt",
            "storm_gen_article.txt",
            "storm_gen_outline.txt"
        ]
        
        for filename in output_files:
            filepath = topic_dir / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding='utf-8')
                    if content.strip():
                        logger.info(f"Using STORM output from: {filename}")
                        return content
                except Exception as e:
                    logger.warning(f"Failed to read {filepath}: {e}")
        
        # Fallback
        logger.warning("No STORM output found")
        return f"# {topic}\n\nNo content generated by STORM"
    
    def run_all_baselines(self, topics: List[str], methods: List[str] = None) -> Dict[str, Any]:
        """Run all specified baselines on topics."""
        if methods is None:
            methods = ["direct_prompting", "storm", "self_rag"]
        
        logger.info(f"Running {len(methods)} methods on {len(topics)} topics")
        
        all_results = {}
        
        for i, topic in enumerate(topics, 1):
            logger.info(f"\nProcessing {i}/{len(topics)}: {topic}")
            topic_results = {}
            
            for method in methods:
                logger.info(f"Running {method}...")
                
                try:
                    if method == "direct_prompting":
                        article = self.run_direct_prompting(topic)
                    elif method == "storm":
                        article = self.run_storm(topic)
                    elif method == "self_rag":
                        article = self.run_self_rag(topic)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    topic_results[method] = {
                        "article": article,
                        "word_count": article.metadata.get("word_count", 0),
                        "success": "error" not in article.metadata
                    }
                    
                    if topic_results[method]["success"]:
                        logger.info(f"✓ {method} completed ({topic_results[method]['word_count']} words)")
                    else:
                        logger.warning(f"✗ {method} failed")
                        
                except Exception as e:
                    logger.error(f"Method {method} failed: {e}")
                    topic_results[method] = {
                        "article": Article(
                            title=topic,
                            content=f"# {topic}\n\nError: {str(e)}",
                            sections={},
                            metadata={"error": str(e), "method": method}
                        ),
                        "word_count": 0,
                        "success": False
                    }
                
                # Small delay between methods
                if method != methods[-1]:
                    time.sleep(2)
            
            all_results[topic] = topic_results
            
            # Delay between topics
            if i < len(topics):
                time.sleep(5)
        
        return all_results