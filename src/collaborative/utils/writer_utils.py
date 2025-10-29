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
) -> str:
    """Build a formatted string of chunk summaries including specified fields."""
    formatted_parts = []
    for query, summary in chunk_summaries.items():
        if max_content_pieces:
            chunks = summary.results[:max_content_pieces]
        else:
            chunks = summary.results

        chunk_strs = []

        for chunk in chunks:
            field_strs = []
            for field in fields:
                if field in chunk and chunk.get(field):
                    val = str(chunk.get(field)).strip()
                    if field == "description" and max_content_pieces is not None:
                        val = val[:100]
                    field_strs.append(f"[{field}: {val}]")
            chunk_strs.append(" ".join(field_strs))

        formatted_parts.append(f"{query}: {{{'; '.join(chunk_strs)}}}")

    return "[" + "; ".join(formatted_parts) + "]"
