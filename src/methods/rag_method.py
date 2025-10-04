# src/methods/rag_method.py
"""
RAG (Retrieval-Augmented Generation) method.
"""

import time

import logging
from typing import List

from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.retrieval.factory import create_retrieval_manager
from src.utils.data import Article
from src.utils.prompts import build_rag_prompt
from src.utils.prompts.templates import build_query_generator_prompt

logger = logging.getLogger(__name__)


class RagMethod(BaseMethod):
    """
    RAG method that retrieves relevant information and uses it to generate articles.

    Workflow:
    1. Generate search queries for the topic
    2. Retrieve relevant passages using retrieval system
    3. Generate article using retrieved context
    """

    def __init__(self):
        super().__init__()
        self.retrieval_config = ConfigContext.get_retrieval_config()

    def run(self, topic: str) -> Article:
        """
        Generate article using RAG approach.

        Args:
            topic: Topic to write about

        Returns:
            Generated article with retrieval metadata
        """
        logger.info(f"Running RAG for: {topic}")

        try:
            start_time = time.time()

            # Get clients and config
            writing_client = ConfigContext.get_client("writing")

            # Create retrieval manager
            retrieval_manager = create_retrieval_manager(
                rm_type=self.retrieval_config.retrieval_manager
            )

            # Generate search queries for the topic
            queries = self._generate_search_queries(writing_client, topic)
            logger.info(f"Generated {len(queries)} search queries for {topic}")

            # Retrieve relevant passages
            all_passages = []
            for query in queries:
                try:
                    results = retrieval_manager.search(
                        query_or_queries=query,
                        max_results=self.retrieval_config.results_per_query,
                    )
                    all_passages.extend(results)
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {e}")

            # Create context from passages
            context = self._create_context_from_passages(all_passages)
            logger.info(f"Retrieved context: {len(context)} characters")

            # Generate article with context
            rag_prompt = build_rag_prompt(topic, context)
            response = writing_client.call_api(rag_prompt)

            # Extract content
            content = response.strip() if response else ""

            generation_time = time.time() - start_time
            content_words = len(content.split()) if content else 0

            # Create article with RAG metadata
            article = Article(
                title=topic,
                content=content,
                sections={},  # RAG method doesn't have structured sections
                metadata={
                    "method": "rag",
                    "generation_time": generation_time,
                    "word_count": content_words,
                    "model": getattr(writing_client, "model_path", "unknown"),
                    "retrieval_manager": self.retrieval_config.retrieval_manager,
                    "num_queries": len(queries),
                    "num_passages": len(all_passages),
                    "context_length": len(context),
                    "queries": queries,
                    "prompt_length": len(rag_prompt),
                    "response_length": len(response) if response else 0,
                },
            )

            logger.info(
                f"RAG completed for {topic} "
                f"({content_words} words, {generation_time:.2f}s, "
                f"{len(all_passages)} passages)"
            )
            return article

        except Exception as e:
            logger.error(f"RAG failed for '{topic}': {e}")
            # Return error article
            return Article(
                title=topic,
                content=f"Error generating article: {str(e)}",
                sections={},
                metadata={
                    "method": "rag",
                    "error": str(e),
                    "generation_time": (
                        time.time() - start_time if "start_time" in locals() else 0
                    ),
                    "word_count": 0,
                },
            )

    def _generate_search_queries(self, client, topic: str) -> List[str]:
        """Generate search queries for the topic."""
        try:
            # Build query generation prompt
            prompt = build_query_generator_prompt(
                topic, num_queries=self.retrieval_config.num_queries
            )

            response = client.call_api(prompt)

            # Parse queries from response (assuming one query per line)
            queries = [
                query.strip()
                for query in response.strip().split("\n")
                if query.strip() and not query.strip().startswith("#")
            ]

            # Ensure we have at least the original topic as a query
            if not queries:
                queries = [topic]
            elif topic not in queries:
                queries.insert(0, topic)

            # Limit to configured number
            return queries[: self.retrieval_config.num_queries]

        except Exception as e:
            logger.warning(f"Query generation failed: {e}, using topic as single query")
            return [topic]

    def _create_context_from_passages(self, passages: List) -> str:
        """Convert search results to context string."""
        if not passages:
            return ""

        context_parts = []
        for i, passage in enumerate(passages[: self.retrieval_config.final_passages]):
            # Handle different passage formats
            if hasattr(passage, "content"):
                content = passage.content
            elif isinstance(passage, dict):
                content = passage.get("content", passage.get("text", str(passage)))
            else:
                content = str(passage)

            # Limit passage length
            if len(content) > self.retrieval_config.passage_max_length:
                content = content[: self.retrieval_config.passage_max_length] + "..."

            context_parts.append(f"Source {i+1}: {content}")

        return "\n\n".join(context_parts)
