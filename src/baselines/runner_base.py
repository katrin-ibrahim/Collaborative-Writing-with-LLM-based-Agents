#!/usr/bin/env python3
"""
Abstract base runner for baseline implementations.
This provides a common interface and shared functionality for both Ollama and local baseline runners.
"""
import time
from abc import ABC, abstractmethod

import logging
from typing import List, Optional

from src.config.baselines_model_config import ModelConfig
from src.utils.baselines_utils import build_direct_prompt, error_article
from src.utils.data_models import Article
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """
    Abstract base runner for all baseline implementations.

    This class defines the common interface that both Ollama and local runners must implement,
    while providing shared functionality to reduce code duplication.
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
        """
        Get the appropriate model engine for the specified task.

        Args:
            task: The task type (e.g., "writing", "critique")

        Returns:
            A model engine instance appropriate for the task
        """

    # ---------------------------------------- Direct Prompting Baseline ----------------------------------------
    def run_direct_prompting(self, topic: str) -> Article:
        """Run direct prompting baseline."""
        logger.info(f"Running Direct Prompting for: {topic}")

        # No need to extract topic string since we now ensure topics are already strings
        prompt = build_direct_prompt(topic)

        try:
            start_time = time.time()

            # Get model engine for writing task
            engine = self.get_model_engine("writing")

            # Generate with engine
            content = self._generate_content(engine, prompt)
            logger.debug(f"Generated content length: {len(content) if content else 0}")

            # Ensure proper title format
            if content and not content.startswith("#"):
                content = f"# {topic}\n\n{content}"

            content_words = len(content.split()) if content else 0
            generation_time = time.time() - start_time

            # Create article with metadata
            article = self._create_article(
                topic, content, "direct", engine, content_words, generation_time
            )

            # Save article if output manager available
            if self.output_manager:
                self.output_manager.save_article(article, "direct")

            logger.info(
                f"Direct Prompting completed for {topic} ({content_words} words, {generation_time:.2f}s)"
            )
            return article

        except Exception as e:
            logger.error(f"Direct prompting failed for '{topic}': {e}")
            return error_article(topic, str(e), "direct")

    def _generate_content(self, engine, prompt: str) -> str:
        """Generate content using the provided engine."""
        # Determine how to extract content based on engine type
        response = engine.generate(
            prompt,
            max_length=self.model_config.get_token_limit_for_task("writing"),
            temperature=self.model_config.get_temperature_for_task("writing"),
        )

        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        elif hasattr(response, "content"):
            return response.content
        else:
            return str(response)

    def _create_article(
        self,
        topic: str,
        content: str,
        method: str,
        engine,
        content_words: int,
        generation_time: float,
    ) -> Article:
        """Create article object with metadata."""
        # Get model name based on engine type
        model_name = getattr(
            engine, "model", getattr(engine, "model_name", str(engine))
        )

        return Article(
            title=topic,
            content=content,
            sections={},
            metadata={
                "method": method,
                "model": model_name,
                "word_count": content_words,
                "generation_time": generation_time,
                "temperature": getattr(engine, "temperature", 0.7),
                "max_tokens": getattr(engine, "max_tokens", 1024),
            },
        )

    # ---------------------------------------- Batch Processing ----------------------------------------
    def run_batch(self, topics: List[str], methods: List[str]) -> List[Article]:
        """Run multiple topics and methods."""
        results = []

        for topic in topics:
            # Filter topics if we have a state manager
            for method in methods:
                if self.state_manager:
                    filtered_topics = self.filter_completed_topics([topic], method)
                    if not filtered_topics:
                        logger.info(
                            f"Topic '{topic}' already completed for method '{method}', skipping"
                        )
                        continue

                # Run appropriate method
                if method == "direct":
                    article = self.run_direct_prompting(topic)
                elif method == "storm":
                    article = (
                        self.run_storm(topic) if hasattr(self, "run_storm") else None
                    )
                elif method == "rag":
                    article = self.run_rag(topic) if hasattr(self, "run_rag") else None
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue

                if article:
                    results.append(article)

                    # Update state if we have a state manager
                    if self.state_manager:
                        self.state_manager.mark_complete(topic, method)

        return results

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
    Main function for running baseline experiments.

    Args:
        args: Command line arguments
        runner_class: The runner class to instantiate
        runner_name: Human-readable name for the runner (for logging)

    Returns:
        0 for success, 1 for failure
    """
    from src.config.baselines_model_config import ModelConfig
    from src.utils.experiment_state_manager import ExperimentStateManager
    from src.utils.output_manager import OutputManager

    try:
        logger.info(f"Starting {runner_name} baseline experiment")
        logger.info(f"Methods: {args.methods}")

        # Create model config
        model_config = ModelConfig(mode=args.backend)

        # Create output manager with proper directory structure
        num_topics = getattr(args, "num_topics", 5)
        output_dir = OutputManager.create_output_dir(
            args.backend, args.methods, num_topics
        )
        output_manager = OutputManager(
            output_dir, debug_mode=args.debug if hasattr(args, "debug") else False
        )

        # Create experiment state manager if resume is enabled
        state_manager = None
        if hasattr(args, "resume") and args.resume:
            logger.info(f"Resuming experiment from checkpoint")
            state_manager = ExperimentStateManager(
                output_dir, create_if_not_exists=True
            )

        # Create runner instance
        runner_kwargs = {}
        if args.backend == "ollama" and hasattr(args, "ollama_host"):
            runner_kwargs["ollama_host"] = args.ollama_host

        runner = runner_class(
            model_config=model_config, output_manager=output_manager, **runner_kwargs
        )

        # Set state manager if available
        if state_manager:
            runner.set_state_manager(state_manager)

        # Get topics from FreshWiki dataset
        try:
            pass

            from src.utils.freshwiki_loader import FreshWikiLoader

            freshwiki = FreshWikiLoader()
            all_topics = [entry.topic for entry in freshwiki.entries]

            if not all_topics:
                logger.error("No topics found in FreshWiki dataset")
                return 1

            # Randomly select num_topics if specified
            num_topics = getattr(args, "num_topics", 5)
            entries = freshwiki.get_evaluation_sample(num_topics)

            # Extract just the topic strings from the FreshWikiEntry objects
            topics = [entry.topic for entry in entries]

            logger.info(f"Extracted {len(topics)} topic strings from FreshWiki entries")

        except Exception as e:
            logger.error(f"Failed to load topics from FreshWiki dataset: {e}")
            return 1

        logger.info(f"Running with {len(topics)} topics")

        # Run the batch
        articles = runner.run_batch(topics, args.methods)
        logger.info(f"Experiment complete. Generated {len(articles)} articles")

        return 0

    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        return 1
