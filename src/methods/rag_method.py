# src/methods/rag_method.py
"""
RAG (Retrieval-Augmented Generation) method.
"""

import hashlib
import time

import logging
from typing import List

from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.retrieval.factory import create_retrieval_manager
from src.utils.data import Article
from src.utils.data.models import ResearchChunk
from src.utils.prompts.templates import build_query_generator_prompt, build_rag_prompt

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

        start_time = time.time()
        try:

            # Get clients and config
            writing_client = ConfigContext.get_client("writing")

            # Create retrieval manager
            retrieval_manager = create_retrieval_manager(
                rm_type=self.retrieval_config.retrieval_manager
            )

            # Generate search queries for the topic
            queries = self._generate_search_queries(writing_client, topic)
            logger.info(f"Generated {len(queries)} search queries for {topic}")

            # Retrieve relevant passages using concurrent search
            all_passages = []
            try:
                # Use concurrent search for better performance with multiple queries
                if self.retrieval_config.retrieval_manager != "faiss":
                    all_results = retrieval_manager.search_concurrent(
                        query_list=queries,
                        max_results=self.retrieval_config.results_per_query,
                    )
                else:
                    # Faiss does not support concurrent search in this implementation
                    all_results = []
                    for query in queries:
                        results = retrieval_manager.search(
                            query_or_queries=query,
                            max_results=self.retrieval_config.results_per_query,
                        )
                        all_results.extend(results)
                # Convert all results to ResearchChunk objects
                # Note: concurrent search returns flattened results, so we create a generic query context
                converted_passages = self._convert_to_research_chunks(
                    all_results,
                    f"Multiple queries: {', '.join(queries[:3])}{'...' if len(queries) > 3 else ''}",
                )
                all_passages.extend(converted_passages)
                logger.info(
                    f"Concurrent search retrieved {len(all_results)} total results"
                )
            except Exception as e:
                logger.warning(
                    f"Concurrent search failed, falling back to sequential: {e}"
                )
                # Fallback to sequential search if concurrent fails
                for query in queries:
                    try:
                        results = retrieval_manager.search(
                            query_or_queries=query,
                            max_results=self.retrieval_config.results_per_query,
                        )
                        converted_passages = self._convert_to_research_chunks(
                            results, query
                        )
                        all_passages.extend(converted_passages)
                    except Exception as seq_e:
                        logger.warning(
                            f"Sequential search failed for query '{query}': {seq_e}"
                        )

            context_passages_count = getattr(
                self.retrieval_config, "max_content_pieces", 10
            )
            context_passages = all_passages[:context_passages_count]

            # Create context from passages
            context = self._create_context_from_passages(context_passages)
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
                    "model": getattr(
                        writing_client,
                        "model_path",
                        getattr(writing_client, "model", "unknown"),
                    ),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        """Convert ResearchChunk objects to context string matching writer_v2 chunk formatting."""
        if not passages:
            return ""

        # Format chunks like writer_v2: {chunk_id} {content: {full_content}, score: {score}, url: {url}}
        context_parts = []
        for passage in passages[: self.retrieval_config.final_passages]:
            # All passages are now ResearchChunk objects
            chunk_id = passage.chunk_id
            content = passage.content
            score = passage.metadata.get("relevance_score", "N/A")
            url = passage.url or "N/A"

            context_parts.append(
                f"{chunk_id} {{content: {content}, score: {score}, url: {url}}}"
            )

        return ", ".join(context_parts)

    def _convert_to_research_chunks(
        self, results: List, query: str
    ) -> List[ResearchChunk]:
        """Convert retrieval results to ResearchChunk objects."""
        research_chunks = []
        rm_type = self.retrieval_config.retrieval_manager

        for i, result in enumerate(results):
            if isinstance(result, dict):
                # Generate chunk ID matching the format used in collaborative tools
                chunk_id = f"search_{rm_type}_{i}_{hashlib.md5(str(result).encode()).hexdigest()[:8]}"
                # Convert using the existing method
                research_chunk = ResearchChunk.from_retrieval_result(chunk_id, result)
                # Add RAG-specific metadata
                research_chunk.metadata.update(
                    {
                        "method": "rag",
                        "search_query": query,
                        "relevance_rank": i + 1,
                        "rm_type": rm_type,
                    }
                )
                research_chunks.append(research_chunk)
            else:
                # If it's already a ResearchChunk or other object, pass it through
                research_chunks.append(result)
        return research_chunks
