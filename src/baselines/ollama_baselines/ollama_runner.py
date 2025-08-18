"""
Ultra-thin Ollama baseline runner - only responsible for client/engine initialization and STORM.
All shared logic moved to BaseRunner.
"""

import logging
from typing import List, Optional

from src.baselines.baseline_runner_base import BaseRunner
from src.baselines.ollama_baselines.ollama_engine import OllamaModelEngine
from src.config.baselines_model_config import ModelConfig
from src.config.retrieval_config import RetrievalConfig
from src.utils.article import error_article
from src.utils.clients import OllamaClient
from src.utils.data import Article
from src.utils.io import OutputManager

# STORM imports - only here to avoid local runner conflicts
from .configure_storm import setup_storm_runner

logger = logging.getLogger(__name__)


class BaselineRunner(BaseRunner):
    """
    Ultra-thin Ollama runner - handles client/engine init + STORM only.
    All other methods (direct, rag, batch) implemented in BaseRunner.
    """

    def __init__(
        self,
        ollama_host: str = None,
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
    ):
        # Initialize base class
        super().__init__(
            model_config=model_config,
            output_manager=output_manager,
            retrieval_config=retrieval_config,
        )

        # Determine ollama_host: model config takes priority over parameter
        if self.model_config and self.model_config.ollama_host:
            self.ollama_host = self.model_config.ollama_host
            logger.info(f"Using ollama_host from model config: {self.ollama_host}")
        elif ollama_host:
            self.ollama_host = ollama_host
            logger.info(f"Using ollama_host from parameter: {self.ollama_host}")
        else:
            self.ollama_host = "http://10.167.31.201:11434/"
            logger.info(f"Using default ollama_host: {self.ollama_host}")

        # Create and verify Ollama client
        self.client = OllamaClient(host=self.ollama_host)
        if not self.client.is_available():
            raise RuntimeError(f"Ollama server not available at {self.ollama_host}")

        # Engine cache
        self._engine_cache = {}

        # Log available models
        available_models = self.client.list_models()
        logger.info(
            f"Connected to Ollama with {len(available_models)} models available"
        )

    def get_model_engine(self, task: str) -> OllamaModelEngine:
        """Get cached OllamaModelEngine for the specified task."""
        if task in self._engine_cache:
            return self._engine_cache[task]

        # Create new engine
        model_path = self.model_config.get_model_path(task)

        engine = OllamaModelEngine(
            ollama_host=self.ollama_host,
            model_path=model_path,
            config=self.model_config,
            task=task,
        )

        # Cache and return
        self._engine_cache[task] = engine
        return engine

    def get_supported_methods(self):
        """Return methods supported by Ollama runner."""
        return [
            "direct",
            "storm",
            "rag",
            "agentic",
            "collaborative",
        ]  # Full method support

    def run_storm(self, topic: str) -> Article:
        """
        Run STORM - only method unique to Ollama runner.
        (direct and rag implemented in BaseRunner)
        """
        logger.info(f"Running STORM for: {topic}")
        import time
        from datetime import datetime

        try:
            start_time = time.time()
            # Setup STORM runner
            storm_runner, storm_output_dir, storm_config = setup_storm_runner(
                client=self.client,
                config=self.model_config,
                retrieval_config=self.retrieval_config,
                storm_output_dir=(
                    self.output_manager.setup_storm_output_dir(topic)
                    if self.output_manager
                    else None
                ),
            )

            # Run STORM
            storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )

            # Extract STORM output and create Article
            from pathlib import Path

            from src.utils.article import extract_storm_output

            content = extract_storm_output(Path(storm_output_dir), topic)

            generation_time = (time.time() - start_time) / 60
            logger.info(
                f"STORM generation time for {topic}: {generation_time:.2f} minutes"
            )

            content_words = len(content.split()) if content else 0

            # Create Article object
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "model": self.model_config.get_model_for_task("writing"),
                    "word_count": content_words,
                    "generation_time": generation_time,
                    "timestamp": datetime.now().isoformat(),
                    "storm_config": storm_config,
                },
            )

            # Save article if output manager available
            if self.output_manager:
                self.output_manager.save_article(article, "storm")

            logger.info(f"STORM completed for {topic}")
            return article

        except Exception as e:
            logger.error(f"STORM failed for {topic}: {e}")
            return error_article(topic, "storm", str(e))

    def run_storm_with_config(self, topic: str, storm_config: dict = None) -> Article:
        """
        Run STORM with custom configuration - needed for optimization.
        """
        logger.info(f"Running STORM with config for: {topic}")
        logger.info(f"Config: {storm_config}")
        import time
        from datetime import datetime

        try:
            start_time = time.time()
            # Setup STORM runner with custom config
            storm_runner, storm_output_dir, final_config = setup_storm_runner(
                client=self.client,
                config=self.model_config,
                retrieval_config=self.retrieval_config,
                storm_output_dir=(
                    self.output_manager.setup_storm_output_dir(topic)
                    if self.output_manager
                    else None
                ),
                storm_config=storm_config,  # Pass custom config
            )

            # Run STORM
            storm_runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )

            # Extract STORM output and create Article
            from pathlib import Path

            from src.utils.article import extract_storm_output

            content = extract_storm_output(Path(storm_output_dir), topic)

            generation_time = (time.time() - start_time) / 60
            logger.info(
                f"STORM generation time for {topic}: {generation_time:.2f} minutes"
            )

            content_words = len(content.split()) if content else 0

            # Create Article object
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "model": self.model_config.get_model_for_task("writing"),
                    "word_count": content_words,
                    "generation_time": generation_time,
                    "timestamp": datetime.now().isoformat(),
                    "storm_config": final_config,
                },
            )

            # Save article if output manager available
            if self.output_manager:
                self.output_manager.save_article(article, "storm")

            logger.info(f"STORM completed for {topic}")
            return article

        except Exception as e:
            logger.error(f"STORM failed for '{topic}': {e}")
            return error_article(topic, str(e), "storm")

    def run_storm_batch(self, topics: List[str], max_workers: int = 1) -> List[Article]:
        """Run STORM in parallel - Ollama-specific implementation."""
        return self._run_batch_generic("storm", self.run_storm, topics, max_workers)

    def _get_query_generator(self):
        def ollama_query_generator(
            engine, topic: str, num_queries: int = 5
        ) -> List[str]:
            prompt = f"""Generate {num_queries} possible Wikipedia article titles related to the topic "{topic}".

            Guidelines:
            - Think about what the article could possibly cover.
            - When you have an idea, what the topic could be about, think of possible sections that could be included in the article.
            - The titles you generate should cover the possible sections of the article.
            - Titles must resemble actual Wikipedia article titles.
            - Use clear, encyclopedic language — avoid vague or overly broad phrases.
            - Include topic-specific terms to ensure relevance.
            - Avoid phrasing like questions, search queries, or casual writing.


            Only output possible Wikipedia article titles — one per line, no numbering or extra text.

            Wikipedia article titles for "{topic}":
            """

            try:
                import re

                # Use engine instead of wrapper for consistency
                engine = self.get_model_engine("fast")
                response = engine.complete(prompt)

                # Extract content using engine's method
                content = engine.extract_content(response)
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

                raw_queries = [
                    line.strip() for line in content.split("\n") if line.strip()
                ]

                # Return raw queries - let BaseRetriever handle all cleaning
                return raw_queries[:num_queries] if raw_queries else [topic]

            except Exception as e:
                logger.error(f"Ollama query generation failed: {e}")
                # Simple fallback
                return [topic]

        return ollama_query_generator

    # All other methods (run_direct, run_rag, run_*_batch) implemented in BaseRunner
