# src/collaborative/tools/tools.py
"""
Core tool definitions for collaborative agents.
"""

import logging
import re
from langchain_core.tools import tool
from typing import Any, Dict

from src.config.retrieval_config import RetrievalConfig
from src.retrieval.factory import create_retrieval_manager

logger = logging.getLogger(__name__)

# Global retrieval manager cache
_retrieval_managers = {}


def _get_retrieval_manager(rm_type: str = "wiki"):
    """Get or create retrieval manager using factory pattern."""
    global _retrieval_managers

    if rm_type not in _retrieval_managers:
        try:
            config = RetrievalConfig()
            config.retrieval_manager_type = rm_type
            _retrieval_managers[rm_type] = create_retrieval_manager(
                retrieval_config=config
            )
            logger.info(f"Created {rm_type} retrieval manager")
        except Exception as e:
            logger.error(f"Failed to create {rm_type} retrieval manager: {e}")
            # Fallback to wiki if other types fail
            if rm_type != "wiki":
                return _get_retrieval_manager("wiki")
            raise

    return _retrieval_managers[rm_type]


@tool
def search_and_retrieve(
    query: str, rm_type: str = "wiki", max_results: int = 5, purpose: str = "general"
) -> Dict[str, Any]:
    """
    Search external sources for information using configurable retrieval manager.

    Args:
        query: Search query string
        rm_type: Type of retrieval manager to use (default: "wiki")
        max_results: Maximum number of results to return
        purpose: Purpose of search ("writing", "fact_checking", "general")

    Returns:
        Dictionary with search results and metadata
    """
    try:
        if not query or not query.strip():
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "summary": "Empty query provided",
                "success": False,
            }

        # Get retrieval manager
        retrieval_manager = _get_retrieval_manager(rm_type)

        # Perform search - RM returns List[Dict] or List[str]
        raw_results = retrieval_manager.search(
            query_or_queries=query.strip(),
            max_results=max_results,
            topic=query,  # Use query as topic for context
        )

        # Format results consistently
        formatted_results = []
        for i, result in enumerate(raw_results[:max_results]):
            if isinstance(result, dict):
                # Result is already a dictionary
                formatted_result = {
                    "content": result.get("content", str(result)),
                    "source": result.get("source", f"{rm_type}_{i+1}"),
                    "relevance_score": result.get("relevance_score", 1.0 - (i * 0.1)),
                    "metadata": result.get(
                        "metadata", {"rm_type": rm_type, "purpose": purpose}
                    ),
                }
            elif isinstance(result, str):
                # Result is a raw string passage
                formatted_result = {
                    "content": result,
                    "source": f"{rm_type}_{i+1}",
                    "relevance_score": 1.0 - (i * 0.1),
                    "metadata": {"rm_type": rm_type, "purpose": purpose},
                }
            else:
                # Handle other formats (e.g., SearchResult objects)
                if hasattr(result, "to_dict"):
                    formatted_result = result.to_dict()
                else:
                    formatted_result = {
                        "content": str(result),
                        "source": f"{rm_type}_{i+1}",
                        "relevance_score": 1.0 - (i * 0.1),
                        "metadata": {"rm_type": rm_type, "purpose": purpose},
                    }

            formatted_results.append(formatted_result)

        logger.info(f"Search completed: '{query}' → {len(formatted_results)} results")

        return {
            "query": query,
            "purpose": purpose,
            "rm_type": rm_type,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "summary": f"Found {len(formatted_results)} results for '{query}'",
            "success": True,
        }

    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return {
            "query": query,
            "purpose": purpose,
            "results": [],
            "total_results": 0,
            "summary": f"Search failed: {e}",
            "success": False,
            "error": str(e),
        }


@tool
def get_article_metrics(content: str, title: str) -> Dict[str, Any]:
    """Get objective metrics about article structure."""
    return {
        "word_count": len(content.split()),
        "heading_count": len(re.findall(r"^#+\\s+(.+)$", content, re.MULTILINE)),
        "paragraph_count": len([p for p in content.split("\\n\\n") if p.strip()]),
    }
