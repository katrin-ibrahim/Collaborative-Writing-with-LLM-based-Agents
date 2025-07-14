# FILE: runners/ollama_runner.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging
from typing import List, Optional

from baselines.configure_storm import setup_storm_runner
from baselines.runner_utils import (
    build_direct_prompt,
    error_article,
    extract_storm_output,
    get_model_wrapper,
)
from config.model_config import ModelConfig
from utils.data_models import Article
from utils.ollama_client import OllamaClient
from utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class BaselineRunner:
    def __init__(
        self,
        ollama_host: str = "http://10.167.31.201:11434/",
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
    ):
        self.ollama_host = ollama_host
        self.model_config = model_config or ModelConfig()
        self.client = OllamaClient(host=ollama_host)
        self.output_manager = output_manager

        if not self.client.is_available():
            raise RuntimeError(f"Ollama server not available at {ollama_host}")

        available_models = self.client.list_models()
        logger.info(
            f"Connected to Ollama with {len(available_models)} models available"
        )

    # ---------------------------------------- Direct Prompting Baseline ----------------------------------------
    def run_direct_prompting(self, topic: str) -> Article:
        logger.info(f"Running Direct Prompting for: {topic}")
        prompt = build_direct_prompt(topic)

        try:
            start_time = time.time()
            wrapper = get_model_wrapper(self.client, self.model_config, "writing")

            response = wrapper(prompt)
            logger.debug(f"Generated response type: {type(response)}")

            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
            elif hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            logger.debug(
                f"Extracted content type: {type(content)}, length: {len(content) if content else 0}"
            )
            logger.debug(
                f"Content preview: {content[:200] if content else 'No content'}..."
            )
            content_words = len(content.split()) if content else 0
            generation_time = time.time() - start_time

            if content and not content.startswith("#"):
                content = f"# {topic}\n\n{content}"

            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct",
                    "model": wrapper.model,
                    "word_count": content_words,
                    "generation_time": generation_time,
                    "temperature": wrapper.temperature,
                },
            )

            if self.output_manager:
                self.output_manager.save_article(article, "direct")

            return article

        except Exception as e:
            logger.error(f"Direct Prompting failed: {e}")
            raise RuntimeError(f"Direct Prompting error for {topic}: {e}")

    def run_direct_batch(
        self, topics: List[str], max_workers: int = 3
    ) -> List[Article]:
        # Filter out completed topics using state manager
        remaining_topics = self.filter_completed_topics(topics, "direct")

        if not remaining_topics:
            logger.info("All direct prompting topics already completed")
            return []

        logger.info(
            f"Running Direct Prompting batch for {len(remaining_topics)} remaining topics"
        )
        results = []

        def run_topic(topic):
            # Mark as in progress at start
            if hasattr(self, "state_manager") and self.state_manager:
                self.state_manager.mark_topic_in_progress(topic, "direct")

            try:
                article = self.run_direct_prompting(topic)
                # Mark as completed on success
                if hasattr(self, "state_manager") and self.state_manager:
                    self.state_manager.mark_topic_completed(topic, "direct")
                return article
            except Exception as e:
                # Clean up in-progress state on failure
                if hasattr(self, "state_manager") and self.state_manager:
                    self.state_manager.cleanup_in_progress_topic(topic, "direct")
                raise e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_topic, topic): topic for topic in remaining_topics
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"Direct batch failed for {topic}: {e}")
                    results.append(error_article(topic, e, "direct_batch"))

        return results

    # ---------------------------------------- STORM Baseline ----------------------------------------
    def run_storm(self, topic: str) -> Article:
        logger.info(f"Running STORM for: {topic}")

        try:
            storm_output_dir = self.output_manager.setup_storm_output_dir(topic)
            runner, storm_output_dir = setup_storm_runner(
                self.client, self.model_config, storm_output_dir
            )

            start_time = time.time()
            runner.run(
                topic=topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )
            generation_time = time.time() - start_time
            content = extract_storm_output(topic, storm_output_dir)

            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "storm",
                    "word_count": len(content.split()) if content else 0,
                    "generation_time": generation_time,
                    "model": self.model_config.get_model_for_task("writing"),
                },
            )

            if self.output_manager:
                self.output_manager.save_article(article, "storm")
                # self.output_manager.cleanup_storm_temp(topic)

            return article

        except Exception as e:
            logger.error(f"STORM failed: {e}")
            raise RuntimeError(f"STORM error for {topic}: {e}")

    def run_storm_batch(self, topics: List[str], max_workers: int = 3) -> List[Article]:
        # Filter out completed topics using state manager
        remaining_topics = self.filter_completed_topics(topics, "storm")

        if not remaining_topics:
            logger.info("All STORM topics already completed")
            return []

        logger.info(f"Running STORM batch for {len(remaining_topics)} remaining topics")
        results = []

        def run_topic(topic):
            # Mark as in progress at start
            if hasattr(self, "state_manager") and self.state_manager:
                self.state_manager.mark_topic_in_progress(topic, "storm")

            try:
                article = self.run_storm(topic)
                # Mark as completed on success
                if hasattr(self, "state_manager") and self.state_manager:
                    self.state_manager.mark_topic_completed(topic, "storm")
                return article
            except Exception as e:
                # Clean up in-progress state on failure
                if hasattr(self, "state_manager") and self.state_manager:
                    self.state_manager.cleanup_in_progress_topic(topic, "storm")
                raise e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_topic, topic): topic for topic in remaining_topics
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"STORM batch failed for {topic}: {e}")
                    results.append(error_article(topic, e, "storm_batch"))

        return results

    # ---------------------------------------- RAG Baseline ----------------------------------------
    def run_rag(self, topic: str) -> Article:
        logger.info(f"Running RAG for: {topic}")

        try:
            from baselines.runner_utils import (
                create_context_from_passages,
                generate_article_with_context,
                generate_search_queries,
                retrieve_and_format_passages,
            )
            from baselines.wikipedia_rm import WikipediaSearchRM

            start_time = time.time()
            retrieval_system = WikipediaSearchRM(k=3)

            # RAG pipeline
            queries = generate_search_queries(self.client, self.model_config, topic)
            passages = retrieve_and_format_passages(retrieval_system, queries)
            context = create_context_from_passages(passages, max_passages=5)
            content = generate_article_with_context(
                self.client, self.model_config, topic, context
            )

            generation_time = time.time() - start_time

            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "rag",
                    "word_count": len(content.split()) if content else 0,
                    "generation_time": generation_time,
                    "model": self.model_config.get_model_for_task("writing"),
                    "queries_used": len(queries),
                    "passages_retrieved": len(passages),
                },
            )

            if self.output_manager:
                self.output_manager.save_article(article, "rag")

            logger.info(f"RAG completed for {topic} in {generation_time:.2f}s")
            return article

        except Exception as e:
            logger.error(f"RAG failed for {topic}: {e}")
            return error_article(topic, e, "rag")

    def run_rag_batch(self, topics: List[str], max_workers: int = 2) -> List[Article]:
        logger.info(f"Running RAG batch for {len(topics)} topics")

        remaining_topics = topics
        if hasattr(self, "state_manager") and self.state_manager:
            remaining_by_method = self.state_manager.get_remaining_topics(topics)
            remaining_topics = remaining_by_method.get("rag", topics)
            completed_count = len(topics) - len(remaining_topics)
            logger.info(
                f"RAG: {completed_count} completed, {len(remaining_topics)} remaining"
            )

        if not remaining_topics:
            logger.info("All RAG topics already completed")
            return []

        results = []

        def run_topic(topic):
            if hasattr(self, "state_manager") and self.state_manager:
                self.state_manager.mark_topic_in_progress(topic, "rag")

            try:
                result = self.run_rag(topic)
                if hasattr(self, "state_manager") and self.state_manager:
                    self.state_manager.mark_topic_completed(topic, "rag")
                return result
            except Exception as e:
                if hasattr(self, "state_manager") and self.state_manager:
                    self.state_manager.cleanup_in_progress_topic(topic, "rag")
                raise e

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_topic, topic): topic for topic in remaining_topics
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    topic = futures[future]
                    logger.error(f"RAG batch failed for {topic}: {e}")
                    results.append(error_article(topic, e, "rag_batch"))

        return results

    # ---------------------------------------- State Management ----------------------------------------

    def set_state_manager(self, state_manager):
        """Set the state manager for checkpoint tracking."""
        self.state_manager = state_manager

    def filter_completed_topics(self, topics: List[str], method: str) -> List[str]:
        """Filter out already completed topics for a method."""
        if not hasattr(self, "state_manager") or not self.state_manager:
            return topics

        remaining = [
            topic
            for topic in topics
            if topic not in self.state_manager.completed_topics.get(method, set())
        ]

        skipped_count = len(topics) - len(remaining)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already completed {method} topics")

        return remaining
