# src/methods/direct_method.py
"""
Direct prompting method - single prompt to generate article.
"""

import time

import logging

from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.utils.data import Article
from src.utils.prompts import build_direct_prompt

logger = logging.getLogger(__name__)


class DirectMethod(BaseMethod):
    """
    Direct prompting method that generates articles using a single comprehensive prompt.

    This is the simplest baseline method that provides a topic to the LLM
    and asks it to generate a full article without any external knowledge.
    """

    def __init__(self):
        super().__init__()

    def run(self, topic: str) -> Article:
        """
        Generate article using direct prompting approach.

        Args:
            topic: Topic to write about

        Returns:
            Generated article with metadata
        """
        logger.info(f"Running direct prompting for: {topic}")

        # Reset usage counters at start
        task_models = self._get_task_models_for_method()
        self._reset_all_client_usage(task_models)

        start_time = time.time()
        try:

            # Get the writing client from ConfigContext
            client = ConfigContext.get_client("writing")

            # Build direct prompt and generate
            prompt = build_direct_prompt(topic)
            response = client.call_api(prompt)

            # Extract content - for direct method, response is the content
            content = response.strip() if response else ""

            generation_time = time.time() - start_time
            content_words = len(content.split()) if content else 0

            # Collect token usage statistics
            token_usage = self._collect_token_usage(task_models)

            # Create article with metadata
            article = Article(
                title=topic,
                content=content,
                sections={},  # Direct method doesn't have structured sections
                metadata={
                    "method": "direct",
                    "generation_time": generation_time,
                    "word_count": content_words,
                    "model": getattr(
                        client, "model_path", getattr(client, "model", "unknown")
                    ),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt_length": len(prompt),
                    "response_length": len(response) if response else 0,
                    "token_usage": token_usage,
                },
            )

            logger.info(
                f"Direct prompting completed for {topic} "
                f"({content_words} words, {generation_time:.2f}s, "
                f"{token_usage['total_tokens']} tokens)"
            )
            return article

        except Exception as e:
            logger.error(f"Direct prompting failed for '{topic}': {e}")
            # Return error article
            generation_time = time.time() - start_time
            return Article(
                title=topic,
                content=f"Error generating article: {str(e)}",
                sections={},
                metadata={
                    "method": "direct",
                    "error": str(e),
                    "generation_time": generation_time,
                    "word_count": 0,
                    "token_usage": self._collect_token_usage(task_models),
                },
            )
