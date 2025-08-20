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
from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG, RetrievalConfig
from src.utils.article import error_article
from src.utils.data import Article, SearchResult
from src.utils.io import OutputManager
from src.utils.prompts import build_direct_prompt, build_rag_prompt

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
        retrieval_config: Optional[RetrievalConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.output_manager = output_manager
        self.retrieval_config = retrieval_config
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
            response = engine.complete(prompt)

            # Extract content using engine's helper method
            content = engine.extract_content(response)

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

            # Use instance config with fallback to default
            config = self.retrieval_config or DEFAULT_RETRIEVAL_CONFIG

            # Generate search queries for the topic
            queries = query_generator(engine, topic, num_queries=config.num_queries)
            # logger.info(f"Generated {len(queries)} search queries for {topic}")

            # Retrieve context
            passages = retrieval_system.search(
                query_or_queries=queries,
                max_results=config.results_per_query,
                topic=topic,
            )
            context = self._create_context_from_passages(passages)

            # Generate article with context
            rag_prompt = build_rag_prompt(topic, context)
            response = engine.complete(rag_prompt)

            # Extract content using engine's helper method
            content = engine.extract_content(response)

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
        self, topics: List[str], max_workers: int = None
    ) -> List[Article]:
        if max_workers is None:
            max_workers = (
                DEFAULT_RETRIEVAL_CONFIG.max_workers_direct
                if len(topics) >= DEFAULT_RETRIEVAL_CONFIG.parallel_threshold
                else 1
            )
        """Run direct prompting in parallel - shared implementation."""
        return self._run_batch_generic("direct", self.run_direct, topics, max_workers)

    def run_rag_batch(
        self, topics: List[str], max_workers: int = None
    ) -> List[Article]:
        """Run RAG in parallel - shared implementation."""
        if max_workers is None:
            max_workers = (
                DEFAULT_RETRIEVAL_CONFIG.max_workers_rag
                if len(topics) >= DEFAULT_RETRIEVAL_CONFIG.parallel_threshold
                else 1
            )
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
            elif method == "rag":
                method_results = self.run_rag_batch(topics)
            elif method == "storm" and hasattr(self, "run_storm_batch"):
                method_results = self.run_storm_batch(topics)
            elif hasattr(self, f"run_{method}"):
                # Fallback: try to call run_{method} for single topics
                method_results = []
                single_method_func = getattr(self, f"run_{method}")
                for topic in topics:
                    try:
                        result = single_method_func(topic)
                        method_results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to run {method} for {topic}: {e}")
                        method_results.append(error_article(topic, str(e), method))
            else:
                logger.warning(f"Method '{method}' not implemented in this runner")
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

        # Only use threading when we have enough topics to benefit
        if len(remaining_topics) < DEFAULT_RETRIEVAL_CONFIG.parallel_threshold:
            # Sequential processing for small batches
            logger.info(f"Processing {len(remaining_topics)} topics sequentially")
            for topic in remaining_topics:
                try:
                    if self.state_manager:
                        self.state_manager.mark_topic_in_progress(topic, method)
                    article = single_method_func(topic)
                    if self.state_manager:
                        self.state_manager.mark_topic_completed(topic, method)
                    results.append(article)
                except Exception as e:
                    if self.state_manager:
                        self.state_manager.cleanup_in_progress_topic(topic, method)
                    logger.error(f"{method} failed for {topic}: {e}")
                    results.append(error_article(topic, str(e), f"{method}"))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_topic, topic): topic
                    for topic in remaining_topics
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
    def _get_retrieval_system(self, retrieval_config=None):
        """
        Get retrieval system using configurable factory.
        Fails clearly instead of providing fallbacks.
        """
        try:
            from src.retrieval import create_retrieval_manager

            # Use provided config, instance config, or default (in that order)
            config = (
                retrieval_config or self.retrieval_config or DEFAULT_RETRIEVAL_CONFIG
            )

            # Create retrieval system using factory - config contains all parameters
            retrieval_system = create_retrieval_manager(retrieval_config=config)

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

    def _create_context_from_passages(self, passages: list[SearchResult]) -> str:

        if not passages:
            raise ValueError("No passages provided for context creation")
        context_parts = []
        for i, passage in enumerate(passages, start=1):
            context_part = (
                f"Context Part {i}:\n \n{passage['snippets']} \n url: {passage['url']}"
            )
            context_parts.append(context_part)

        return "\n\n ".join(context_parts)

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

        # Get model name safely, avoiding the actual model object
        model_name = None
        if hasattr(engine, "model_name"):
            model_name = engine.model_name
        elif hasattr(engine, "model") and isinstance(engine.model, str):
            model_name = engine.model
        else:
            model_name = str(engine)

        metadata = {
            "method": method,
            "model": model_name,
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
    from src.utils.data import FreshWikiLoader
    from src.utils.experiment import ExperimentStateManager, setup_output_directory
    from src.utils.io import setup_logging

    setup_logging(args.log_level)

    try:
        # Create retrieval configuration with CLI overrides
        retrieval_config = None
        if hasattr(args, "retrieval_manager") and args.retrieval_manager:
            # Use hybrid approach: base RM config + CLI overrides
            try:
                from src.config.retrieval_config import RetrievalConfig

                # Collect CLI overrides
                overrides = {}
                if hasattr(args, "semantic_filtering") and args.semantic_filtering:
                    overrides["semantic_filtering_enabled"] = (
                        args.semantic_filtering.lower() == "true"
                    )

                retrieval_config = RetrievalConfig.from_base_config_with_overrides(
                    rm_type=args.retrieval_manager, **overrides
                )

            except Exception as e:
                logger.error(f"Failed to create hybrid retrieval config: {e}")
                logger.info("Falling back to default retrieval configuration")
                retrieval_config = None
        else:
            # Use default config with CLI overrides
            try:
                from dataclasses import replace

                from src.config.retrieval_config import DEFAULT_RETRIEVAL_CONFIG

                retrieval_config = DEFAULT_RETRIEVAL_CONFIG

                # Apply CLI overrides if any
                if hasattr(args, "semantic_filtering") and args.semantic_filtering:
                    semantic_enabled = args.semantic_filtering.lower() == "true"
                    retrieval_config = replace(
                        retrieval_config, semantic_filtering_enabled=semantic_enabled
                    )
                    logger.info(
                        f"🔍 Semantic filtering {'enabled' if semantic_enabled else 'disabled'} from CLI"
                    )

            except Exception as e:
                logger.error(f"Failed to apply CLI overrides to default config: {e}")
                retrieval_config = None

        # Load model configuration with hybrid approach
        if hasattr(args, "model_config") and args.model_config in [
            "ollama_localhost",
            "ollama_ukp",
            "slurm",
            "slurm_thinking",
        ]:
            try:
                model_config = ModelConfig.from_yaml(args.model_config)
                if args.override_model:
                    model_config.override_model = args.override_model
                logger.info(f"📋 Loaded model config: {args.model_config}")
                logger.info(f"🎯 Model mode: {model_config.mode}")
            except Exception as e:
                logger.error(f"Failed to load model config '{args.model_config}': {e}")
                logger.info("Falling back to default model configuration")
                model_config = ModelConfig(
                    mode=args.backend, override_model=args.override_model
                )
        else:
            # Default model config
            model_config = ModelConfig(
                mode=args.backend, override_model=args.override_model
            )

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

        runner = runner_class(
            model_config=model_config,
            output_manager=output_manager,
            retrieval_config=retrieval_config,
            **runner_kwargs,
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
