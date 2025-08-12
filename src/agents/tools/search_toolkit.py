# src/agents/tools/search_toolkit.py
from langchain_core.tools import tool
from typing import Any, Dict

from src.retrieval.wiki_rm import WikiRM
from src.utils.data_models import SearchResult

# Global retrieval manager instance for tools
_retrieval_manager = None


def _get_retrieval_manager():
    """Get or create retrieval manager instance."""
    global _retrieval_manager
    if _retrieval_manager is None:
        _retrieval_manager = WikiRM(max_articles=5, max_sections=10)
    return _retrieval_manager


@tool
def search_all_sources(
    query: str, wiki_results: int = 3, web_results: int = 3
) -> Dict[str, Any]:
    """
    Search Wikipedia sources for comprehensive information.

    Args:
        query: The search query string
        wiki_results: Maximum Wikipedia results (default: 3)
        web_results: Maximum web results (default: 3) - currently only Wikipedia supported

    Returns:
        Dictionary with search results and metadata for LLM processing
    """
    rm = _get_retrieval_manager()

    # Use the retrieval manager to search with topic context
    passages = rm.search(
        queries=[query],
        max_results=wiki_results + web_results,  # Total results requested
        format_type="rag",
        topic=query,  # Pass query as topic for better targeting
    )

    # Convert passages to SearchResult format with better source identification
    search_results = []
    for i, passage in enumerate(passages):
        # Try to extract more meaningful source info
        source_name = f"Wikipedia"
        if "written in all major" in passage.lower():
            source_name = "Wikipedia_Music_Keys"
        elif "bach" in passage.lower() or "chopin" in passage.lower():
            source_name = "Wikipedia_Classical_Music"
        else:
            source_name = f"Wikipedia_{i+1}"

        search_result = SearchResult(
            content=passage,
            source=source_name,
            relevance_score=1.0 - (i * 0.05),  # Slower decreasing relevance
        )
        search_results.append(search_result)

    return {
        "query": query,
        "results_count": len(search_results),
        "results": [result.to_dict() for result in search_results],
        "summary": f"Found {len(search_results)} Wikipedia results for '{query}'",
    }
