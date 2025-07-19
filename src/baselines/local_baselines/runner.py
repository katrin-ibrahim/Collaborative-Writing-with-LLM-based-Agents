# FILE: local_baselines/runner.py

import sys
import time
from pathlib import Path

import logging
from typing import List, Optional

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.local_baselines.model_engine import LocalModelEngine

from src.config.baselines_model_config import ModelConfig
from src.utils.baselines_utils import (
    build_direct_prompt,
    error_article,
)
from src.utils.data_models import Article
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class LocalBaselineRunner:
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
        device: str = "auto",
    ):
        # Initialize with explicit local mode if not already set
        self.state_manager = None
        self.model_config = model_config or ModelConfig()
        self.model_config.mode = "local"

        self.output_manager = output_manager
        self.device = device

        # Get model path for the writing task
        self.model_path = self.model_config.get_model_path(task="writing")

        # Initialize the engine with the proper parameters
        self.engine = LocalModelEngine(
            model_path=self.model_path,
            device=self.device,
            config=self.model_config,
            task="writing",
        )
        self.output_manager = output_manager

        available_models = self.engine.list_available_models()
        logger.info(
            f"LocalBaselineRunner initialized with model: {self.model_path} and available device: {available_models}"
        )
        logger.info(f"Device: {self.device}")

    # ---------------------------------------- Direct Prompting Baseline ----------------------------------------
    def run_direct_prompting(self, topic: str) -> Article:
        logger.info(f"Running Optimized Local Direct Prompting for: {topic}")

        prompt = build_direct_prompt(topic)

        try:
            start_time = time.time()
            response = self.engine.generate(
                prompt,
                max_length=1024,
                temperature=0.3,
            )

            content = response
            logger.debug(f"Generated content length: {len(content)}")

            # if content and not content.startswith("#"):
            #     content = f"# {topic}\n\n{content}"

            content_words = len(content.split()) if content else 0
            generation_time = time.time() - start_time

            # Create article with same structure as Ollama baselines
            article = Article(
                title=topic,
                content=content,
                sections={},
                metadata={
                    "method": "direct",
                    "model": str(self.model_path),
                    "word_count": content_words,
                    "generation_time": generation_time,
                    "temperature": 0.3,  # TODO: This should be from the model wrapper
                    "max_tokens": 1024,
                },
            )

            if self.output_manager:
                self.output_manager.save_article(article, "direct")

            logger.info(
                f"Direct Prompting completed for {topic} ({content_words} words, {generation_time:.2f}s)"
            )
            return article

        except Exception as e:
            logger.error(f"Direct prompting failed for '{topic}': {e}")
            return error_article(topic, str(e), "direct")

    # ---------------------------------------- STORM Baseline ----------------------------------------

    # ---------------------------------------- RAG Baseline ----------------------------------------

    # ---------------------------------------- Batch Processing ----------------------------------------
    def run_batch(self, topics: List[str], methods: List[str]) -> List[Article]:
        """Run multiple topics and methods efficiently."""
        results = []

        for topic in topics:
            for method in methods:
                if method == "direct":
                    article = self.run_direct_prompting(topic)
                else:
                    logger.warning(f"Unknown method: {method}")
                    continue

                results.append(article)

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
