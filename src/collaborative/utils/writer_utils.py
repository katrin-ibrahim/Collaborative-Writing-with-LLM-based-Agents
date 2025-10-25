from typing import Any, Dict, List

from src.collaborative.memory.memory import SharedMemory
from src.utils.data.models import ResearchChunk  # We'll still need this


def build_full_article_content(title: str, sections: Dict[str, str]) -> str:
    """
    Build the full article content from a title and sections using markdown headings.
    Args:
        title: The article title (will be H1).
        sections: Dict mapping section names to content (each will be H2).
    Returns:
        Concatenated article string with title and sections.
    """
    parts = [f"# {title}\n"]
    for section, content in sections.items():
        parts.append(f"## {section}\n{content.strip()}\n")
    return "\n".join(parts).strip()


def build_formatted_chunk_summaries(
    chunk_summaries: Dict[str, Any], max_content_pieces: int, fields: List[str]
) -> str:
    """Build a formatted string of chunk summaries including specified fields."""
    formatted_parts = []
    for query, summary in chunk_summaries.items():
        chunks = summary.get("chunk_summaries", [])[:max_content_pieces]
        chunk_strs = []
        for chunk in chunks:
            parts = [chunk.get(field, "").strip() for field in fields if field in chunk]
            chunk_strs.append(" | ".join(parts))
        formatted_parts.append(f"{query}: {{{', '.join(chunk_strs)}}}")
    return "[" + "; ".join(formatted_parts) + "]"


def process_and_store_chunks(
    search_id: str,
    source_queries: List[str],
    chunks: List[ResearchChunk],
    rm_type: str,
    shared_memory: SharedMemory,
) -> Dict[str, Any]:
    """
    Processes a list of ResearchChunks retrieved by the RM (potentially from a concurrent search),
    stores them in shared memory, and returns the structured search summary.

    This function is the D.R.Y. implementation that replaces the old search_and_retrieve tool logic.
    """
    if not chunks:
        # Create a summary for a successful search that returned no results
        search_result = {
            "source_queries": source_queries,
            "rm_type": rm_type,
            "total_chunks": 0,
            "results": [],  # Using 'results' key to match SearchSummary Pydantic model
            "metadata": {"message": f"No chunks found for search_id: {search_id}"},
            "success": True,
        }
        shared_memory.store_search_summary(search_id, search_result)
        return search_result

    # The RM has already handled chunk creation, deduplication, and ranking.
    # 1. Store chunks and get the required summaries
    # NOTE: The store_research_chunks method returns the list of summary objects/dicts needed for SearchSummary.results
    chunk_summaries = shared_memory.store_chunks(chunks)

    # 2. Build the final search result summary dictionary
    search_result = {
        "source_queries": source_queries,
        "rm_type": rm_type,
        "total_chunks": len(chunk_summaries),
        "results": chunk_summaries,
        "metadata": {
            "message": f"Found and stored {len(chunk_summaries)} unique chunks.",
            "source_rm": rm_type,
        },
        "success": True,
    }

    # 3. Store the search summary in memory using the unique search_id
    shared_memory.store_search_summary(search_id, search_result)

    return search_result
