"""
Unified runner implementation for RAG functionality.
This provides RAG capabilities for both Ollama and Local runners.
"""

import time

import logging
from typing import Any, Optional

from src.utils.baselines_utils import build_rag_prompt, error_article
from src.utils.data_models import Article

logger = logging.getLogger(__name__)


def run_rag(
    engine: Any,
    topic: str,
    context_retriever: Optional[Any] = None,
    generate_queries_func: Optional[Any] = None,
    num_queries: int = 5,
    max_passages: int = 3,
) -> Article:
    """
    Run RAG pipeline with configurable components.
    This implementation can be used by both Ollama and Local runners.

    Args:
        engine: The model engine (Ollama or Local)
        topic: The topic to generate an article about
        context_retriever: A retrieval system with search method
        generate_queries_func: Function to generate search queries
        num_queries: Number of queries to generate
        max_passages: Maximum number of passages to include in context

    Returns:
        Generated Article object
    """
    logger.info(f"Running RAG for: {topic}")

    try:
        start_time = time.time()

        # Step 1: Generate search queries
        if generate_queries_func:
            # Use provided query generation function
            queries = generate_queries_func(engine, topic, num_queries)
        else:
            raise ValueError(
                "No query generation function provided. Please specify one."
            )

        # Step 2: Retrieve information
        if context_retriever:
            # Use actual retrieval system
            passages = context_retriever.search(queries, max_results=max_passages)
            # Format passages into context
            context = "\n\n".join(
                [f"[Source {i+1}]: {p}" for i, p in enumerate(passages)]
            )
        else:
            raise ValueError(
                "No context retriever provided. Please specify a retrieval system."
            )

        logger.info(f"Created context with {len(context)} characters for {topic}")

        # Step 3: Generate article with retrieved context
        rag_prompt = build_rag_prompt(topic, context)
        # TODO: Hardcode max_length and temperature for RAG generation
        content = engine.generate(rag_prompt, max_length=1024, temperature=0.3)

        generation_time = time.time() - start_time
        content_words = len(content.split()) if content else 0

        # Create article
        article = Article(
            title=topic,
            content=content,
            sections={},
            metadata={
                "method": "rag",
                "model": getattr(engine, "model_name", str(engine)),
                "word_count": content_words,
                "generation_time": generation_time,
                "num_queries": len(queries),
                "context_length": len(context),
            },
        )

        logger.info(
            f"RAG completed for {topic} ({content_words} words, {generation_time:.2f}s)"
        )
        return article

    except Exception as e:
        logger.error(f"RAG failed for '{topic}': {e}")
        return error_article(topic, str(e), "rag")
