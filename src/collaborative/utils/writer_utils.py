from typing import Dict, List, Optional

from src.collaborative.utils.models import SearchSummary


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
    chunk_summaries: Dict[str, SearchSummary],
    max_content_pieces: Optional[int],
    fields: List[str],
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> str:
    """
    Build a formatted string of chunk summaries including specified fields.
    Always flattens chunks across all search summaries for simpler, more compact output.
    Chunks are sorted by relevance (score if available, otherwise rank) before limiting/slicing.

    Args:
        chunk_summaries: Dictionary mapping search IDs to SearchSummary objects
        max_content_pieces: Max total chunks to include (applied after flattening, before slicing)
        fields: List of fields to include in formatting (e.g., ["description", "chunk_id"])
        start_idx: Starting index for chunk slicing (for batching; overrides max_content_pieces)
        end_idx: Ending index for chunk slicing (for batching; overrides max_content_pieces)

    Returns:
        Formatted string representation of chunks (semicolon-separated)
    """
    all_chunks = []
    for summary in chunk_summaries.values():
        all_chunks.extend(summary.results)

    # Sort by relevance: prioritize relevance_score if available, otherwise use relevance_rank
    all_chunks.sort(
        key=lambda c: (
            -c.get("relevance_score", 0.0)
            if c.get("relevance_score") is not None
            else c.get("relevance_rank", float("inf"))
        )
    )

    # Apply max_content_pieces to limit total chunks (unless using explicit slicing)
    if start_idx is None and end_idx is None and max_content_pieces is not None:
        all_chunks = all_chunks[:max_content_pieces]
    elif start_idx is not None and end_idx is not None:
        all_chunks = all_chunks[start_idx:end_idx]

    chunk_strs = []
    for chunk in all_chunks:
        field_strs = []
        for field in fields:
            if field in chunk and chunk.get(field):
                val = str(chunk.get(field)).strip()
                field_strs.append(f"[{field}: {val}]")
        chunk_strs.append(" ".join(field_strs))

    return "; ".join(chunk_strs)
