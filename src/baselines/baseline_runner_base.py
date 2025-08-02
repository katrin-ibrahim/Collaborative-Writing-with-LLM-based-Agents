"""
BaseRunner with all shared logic - ultra-clean architecture.
Concrete runners only handle engine/client initialization.
"""

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
from typing import List, Optional

from src.config.baselines_model_config import ModelConfig
from src.utils.baselines_utils import (
    build_direct_prompt,
    build_rag_prompt,
    error_article,
)
from src.utils.data_models import Article
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """
    Base runner containing ALL shared logic for baseline implementations.
    Concrete runners only handle engine/client initialization and STORM.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.output_manager = output_manager
        self.state_manager = None

    @abstractmethod
    def get_model_engine(self, task: str):
        """Get the appropriate model engine for the specified task."""

    @abstractmethod
    def get_supported_methods(self) -> List[str]:
        """Return list of methods supported by this runner."""

    # ---------------------------------------- Direct Prompting ----------------------------------------
    def run_direct(self, topic: str) -> Article:
        """Run direct prompting baseline - shared implementation."""
        logger.info(f"Running direct prompting for: {topic}")

        try:
            start_time = time.time()

            # Get writing engine
            engine = self.get_model_engine("writing")

            # Build prompt and generate
            prompt = build_direct_prompt(topic)
            content = engine.generate(prompt)

            generation_time = time.time() - start_time
            content_words = len(content.split()) if content else 0

            # Create article
            article = self._create_article(
                topic=topic,
                content=content,
                method="direct",
                engine=engine,
                generation_time=generation_time,
                word_count=content_words,
            )

            # Save article
            if self.output_manager:
                self.output_manager.save_article(article, "direct")

            logger.info(
                f"Direct prompting completed for {topic} ({content_words} words, {generation_time:.2f}s)"
            )
            return article

        except Exception as e:
            logger.error(f"Direct prompting failed for '{topic}': {e}")
            return error_article(topic, str(e), "direct")

    # ---------------------------------------- RAG Implementation ----------------------------------------
    def run_rag(self, topic: str) -> Article:
        """Run RAG baseline - shared implementation with retrieval abstraction."""
        logger.info(f"Running RAG for: {topic}")

        try:
            start_time = time.time()

            # Get writing engine
            engine = self.get_model_engine("writing")

            # Get retrieval system and query generator (runner-specific)
            retrieval_system = self._get_retrieval_system()
            query_generator = self._get_query_generator()

            # Generate search queries
            queries = query_generator(engine, topic, num_queries=5)
            logger.info(f"Generated {len(queries)} search queries for {topic}")

            # Retrieve context
            passages = retrieval_system.search(queries, max_results=8)
            context = self._create_context_from_passages(passages)
            logger.info(f"Created context with {len(context)} characters for {topic}")

            # Generate article with context
            rag_prompt = build_rag_prompt(topic, context)
            content = engine.generate(rag_prompt, max_length=1024, temperature=0.3)

            generation_time = time.time() - start_time
            content_words = len(content.split()) if content else 0

            # Create article
            article = self._create_article(
                topic=topic,
                content=content,
                method="rag",
                engine=engine,
                generation_time=generation_time,
                word_count=content_words,
                extra_metadata={
                    "num_queries": len(queries),
                    "context_length": len(context),
                },
            )

            # Save article
            if self.output_manager:
                self.output_manager.save_article(article, "rag")

            logger.info(
                f"RAG completed for {topic} ({content_words} words, {generation_time:.2f}s)"
            )
            return article

        except Exception as e:
            logger.error(f"RAG failed for '{topic}': {e}")
            return error_article(topic, str(e), "rag")

    # ---------------------------------------- Batch Processing ----------------------------------------
    def run_direct_batch(
        self, topics: List[str], max_workers: int = 2
    ) -> List[Article]:
        """Run direct prompting in parallel - shared implementation."""
        return self._run_batch_generic("direct", self.run_direct, topics, max_workers)

    def run_storm_batch(self, topics: List[str], max_workers: int = 1) -> List[Article]:
        """Run STORM in parallel - shared implementation."""
        if not hasattr(self, "run_storm"):
            logger.warning("STORM not supported by this runner")
            return []
        return self._run_batch_generic("storm", self.run_storm, topics, max_workers)

    def run_rag_batch(self, topics: List[str], max_workers: int = 2) -> List[Article]:
        """Run RAG in parallel - shared implementation."""
        return self._run_batch_generic("rag", self.run_rag, topics, max_workers)

    def run_batch(self, topics: List[str], methods: List[str]) -> List[Article]:
        """Run multiple topics and methods - shared orchestration."""
        results = []
        supported_methods = self.get_supported_methods()

        for method in methods:
            if method not in supported_methods:
                logger.warning(
                    f"Method '{method}' not supported by this runner. Supported: {supported_methods}"
                )
                continue

            # Run method
            if method == "direct":
                method_results = self.run_direct_batch(topics)
            elif method == "storm":
                method_results = self.run_storm_batch(topics)
            elif method == "rag":
                method_results = self.run_rag_batch(topics)
            else:
                logger.warning(f"Unknown method: {method}")
                continue

            results.extend(method_results)

        return results

    # ---------------------------------------- Generic Batch Pattern ----------------------------------------
    def _run_batch_generic(
        self, method: str, single_method_func, topics: List[str], max_workers: int
    ) -> List[Article]:
        """Generic batch processing pattern - eliminates all duplicate batch code."""
        # Filter completed topics
        remaining_topics = self.filter_completed_topics(topics, method)
        if not remaining_topics:
            logger.info(f"All {method} topics already completed")
            return []

        results = []

        def run_topic(topic):
            try:
                if self.state_manager:
                    self.state_manager.mark_topic_in_progress(topic, method)

                article = single_method_func(topic)

                if self.state_manager:
                    self.state_manager.mark_topic_completed(topic, method)

                return article

            except Exception as e:
                if self.state_manager:
                    self.state_manager.cleanup_in_progress_topic(topic, method)
                logger.error(f"{method} batch failed for {topic}: {e}")
                return error_article(topic, str(e), f"{method}_batch")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_topic, topic): topic for topic in remaining_topics
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"{method} batch failed for {topic}: {e}")
                    results.append(error_article(topic, str(e), f"{method}_batch"))

        return results

    # ---------------------------------------- Retrieval Abstraction ----------------------------------------
    def _get_retrieval_system(self):
        """
        Get retrieval system.
        Fails clearly instead of providing fallbacks.
        """
        try:
            from src.retrieval.rm import RetrievalFactory

            # Create Wikipedia-based RM
            retrieval_system = RetrievalFactory.create_wikipedia_rm(
                max_articles=3, max_sections=3
            )

            logger.info(f"Created retrieval system: {retrieval_system.get_stats()}")
            return retrieval_system

        except ImportError as e:
            raise RuntimeError(f"Failed to import retrieval system: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to create retrieval system: {e}") from e

    @abstractmethod
    def _get_query_generator(self):
        """
        Get query generation function - backend specific.
        """

    def _create_context_from_passages(
        self, passages: List[str], max_passages: int = 8
    ) -> str:
        """
        Create context from passages - simplified unified implementation.
        """
        if not passages:
            raise ValueError("No passages provided for context creation")

        # Take top passages and format cleanly
        top_passages = passages[:max_passages]

        context_parts = []
        for i, passage in enumerate(top_passages, 1):
            if passage and len(passage.strip()) > 0:
                # Clean up the passage
                cleaned_passage = passage.strip()

                # Limit length to prevent overwhelming context
                if len(cleaned_passage) > 800:
                    cleaned_passage = cleaned_passage[:800] + "..."

                context_parts.append(f"[Source {i}]: {cleaned_passage}")

        if not context_parts:
            raise ValueError("No valid passages found after cleaning")

        context = "\n\n".join(context_parts)
        logger.debug(
            f"Created context with {len(context_parts)} passages, {len(context)} characters"
        )

        return context

    # ---------------------------------------- Utilities ----------------------------------------
    def _create_article(
        self,
        topic: str,
        content: str,
        method: str,
        engine,
        generation_time: float,
        word_count: int,
        extra_metadata: dict = None,
    ) -> Article:
        """Create Article object - shared implementation."""
        metadata = {
            "method": method,
            "model": getattr(
                engine, "model", getattr(engine, "model_name", str(engine))
            ),
            "word_count": word_count,
            "generation_time": generation_time,
            "temperature": getattr(engine, "temperature", 0.7),
            "max_tokens": getattr(engine, "max_tokens", 1024),
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        return Article(
            title=topic,
            content=content,
            sections={},
            metadata=metadata,
        )

    # ---------------------------------------- State Management ----------------------------------------
    def set_state_manager(self, state_manager):
        """Set the experiment state manager."""
        self.state_manager = state_manager

    def filter_completed_topics(self, topics: List[str], method: str) -> List[str]:
        """Filter out completed topics for the specified method."""
        if not self.state_manager:
            return topics

        return [
            topic
            for topic in topics
            if not self.state_manager.is_complete(topic, method)
        ]


# ---------------------------------------- Main Experiment Function ----------------------------------------
def run_baseline_experiment(args, runner_class, runner_name):
    """
    Single canonical experiment runner - eliminates duplication.
    """
    from src.utils.baselines_utils import setup_output_directory
    from src.utils.experiment_state_manager import ExperimentStateManager
    from src.utils.freshwiki_loader import FreshWikiLoader
    from src.utils.logging_setup import setup_logging

    setup_logging(args.log_level)
    logger.info(f"🔬 {runner_name} Baseline Experiment Runner")
    logger.info(f"📝 Topics: {args.num_topics}")

    try:
        # Load configuration
        model_config = ModelConfig(mode=args.backend)

        # Setup output directory
        output_dir = setup_output_directory(args)
        output_manager = OutputManager(
            str(output_dir), debug_mode=getattr(args, "debug", False)
        )

        # Initialize state manager
        state_manager = ExperimentStateManager(output_dir, args.methods)
        is_resume = (
            state_manager.load_checkpoint()
            if hasattr(state_manager, "load_checkpoint")
            else False
        )

        if is_resume:
            logger.info("🔄 Resuming experiment from checkpoint")
        else:
            logger.info("🆕 Starting new experiment")

        # Create runner instance
        runner_kwargs = {}
        if runner_name.lower() == "ollama" and hasattr(args, "ollama_host"):
            runner_kwargs["ollama_host"] = args.ollama_host
        elif runner_name.lower() == "local":
            if hasattr(args, "device"):
                runner_kwargs["device"] = args.device
            if hasattr(args, "model_path"):
                runner_kwargs["model_path"] = args.model_path

        runner = runner_class(
            model_config=model_config, output_manager=output_manager, **runner_kwargs
        )

        runner.set_state_manager(state_manager)

        # Load topics
        freshwiki = FreshWikiLoader()
        entries = freshwiki.load_topics(args.num_topics)
        if not entries:
            logger.error("No FreshWiki entries found!")
            return 1

        topics = [entry.topic for entry in entries]
        logger.info(f"✅ Loaded {len(topics)} topics")

        # Run experiments
        logger.info("🚀 Starting processing...")
        start_time = time.time()

        # Validate methods against runner capabilities
        supported_methods = runner.get_supported_methods()
        valid_methods = [m for m in args.methods if m in supported_methods]

        # Run batch processing
        results = runner.run_batch(topics, valid_methods)

        # Log completion
        total_time = time.time() - start_time
        total_time = total_time / 60  # Convert to minutes
        logger.info(f"🎉 Experiment completed in {total_time:2f}s")
        logger.info(f"📊 Generated {len(results)} articles")

        return 0

    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        return 1
