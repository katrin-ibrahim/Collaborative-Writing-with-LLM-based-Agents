import logging
import re
from typing import Any, Dict, List

from src.collaborative.memory.memory import SharedMemory
from src.collaborative.utils.models import (
    FeedbackStatus,
    FeedbackStoredModel,
    ReviewerTaskValidationModel,
)
from src.utils.data.models import Article

logger = logging.getLogger(__name__)


def get_article_metrics(article_content: str) -> Dict[str, Any]:
    """Get objective metrics about article structure."""
    try:
        metrics = {
            "word_count": len(article_content.split()),
            "heading_count": len(
                re.findall(r"^#+\s+(.+)$", article_content, re.MULTILINE)
            ),
            "paragraph_count": len(
                [p for p in article_content.split("\n\n") if p.strip()]
            ),
        }

        return {"success": True, "metrics": metrics}
    except Exception as e:
        return {"success": False, "error": str(e), "metrics": {}}


def extract_citations(article: Article) -> List[Dict]:
    """Extract all <c cite="..."/> tags from article text."""
    structured_claims = []
    citation_pattern = r'<c cite="([^"]+)"/>'

    sections = article.sections if article.sections else {"General": article.content}

    for section_name, section_content in sections.items():
        sentences = split_into_sentences(section_content)

        for sentence in sentences:
            matches = re.findall(citation_pattern, sentence)
            if matches:
                clean_sentence = re.sub(citation_pattern, "", sentence).strip()
                structured_claims.append(
                    {
                        "section": section_name,
                        "sentence": clean_sentence,
                        "chunks": matches,
                        "original_sentence": sentence,
                    }
                )

    return structured_claims


def build_ref_map(structured_claims: List[Dict]) -> Dict[str, int]:
    """Build mapping of chunk IDs to reference numbers."""
    ref_map = {}
    ref_counter = 1

    for claim in structured_claims:
        for chunk_id in claim["chunks"]:
            if chunk_id not in ref_map:
                ref_map[chunk_id] = ref_counter
                ref_counter += 1

    return ref_map


def validate_citations(
    structured_claims: List[Dict], ref_map: Dict[str, int], shared_memory: SharedMemory
) -> Dict:
    """Validate citations and flag issues."""
    validation_results = {
        "total_citations": len(ref_map),
        "valid_citations": 0,
        "missing_chunks": [],
        "needs_source_count": 0,
        "duplicate_citations": 0,
    }

    # Check for needs_source tags
    article_content = shared_memory.get_current_draft()
    needs_source_matches = re.findall(r"<needs_source/>", article_content)
    validation_results["needs_source_count"] = len(needs_source_matches)

    # Validate chunk existence
    try:
        chunks_data = shared_memory.get_chunks_by_ids(list(ref_map.keys()))
        found_chunk_ids = set(chunks_data.keys())

        for chunk_id in ref_map.keys():
            if chunk_id in found_chunk_ids:
                validation_results["valid_citations"] += 1
            else:
                validation_results["missing_chunks"].append(chunk_id)
    except Exception as e:
        logger.warning(f"Citation validation failed: {e}")

    # Check for duplicates
    chunk_counts = {}
    for claim in structured_claims:
        for chunk_id in claim["chunks"]:
            chunk_counts[chunk_id] = chunk_counts.get(chunk_id, 0) + 1

    validation_results["duplicate_citations"] = sum(
        1 for count in chunk_counts.values() if count > 1
    )

    return validation_results


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting (can be improved with nltk)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def slug(s: str) -> str:
    return (
        re.sub(r"[^a-zA-Z0-9]+", "-", (s or "").strip().lower()).strip("-") or "section"
    )


def finalize_feedback_items(
    validation: ReviewerTaskValidationModel, iteration: int, max_items: int = 10
) -> list[FeedbackStoredModel]:
    """
    Build IDs + set status='pending'. Limits to max_items by prioritizing high priority feedback.
    Validates that section names are real (not '_overall' or 'N/A').
    in:  ReviewerTaskValidationModel (LLM items without id/status)
    out: List[FeedbackStoredModel] (ready to store, limited to max_items)
    """
    # Filter out invalid section names
    valid_items = []
    invalid_count = 0
    for item in validation.items:
        if item.section in ("_overall", "N/A", "n/a", "NA", "na"):
            invalid_count += 1
            logger.warning(
                f"Skipping feedback item with invalid section name '{item.section}': {item.issue[:50]}..."
            )
            continue
        valid_items.append(item)

    if invalid_count > 0:
        logger.warning(
            f"Filtered out {invalid_count} feedback items with invalid section names "
            f"(must use actual section names, not '_overall' or 'N/A')"
        )

    # Sort by priority: high > medium > low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    sorted_items = sorted(valid_items, key=lambda x: priority_order.get(x.priority, 3))

    # Take only the top max_items
    items_to_process = sorted_items[:max_items]

    if len(valid_items) > max_items:
        logger.info(
            f"Limiting feedback from {len(valid_items)} to {max_items} items "
            f"(keeping {sum(1 for x in items_to_process if x.priority == 'high')} high, "
            f"{sum(1 for x in items_to_process if x.priority == 'medium')} medium, "
            f"{sum(1 for x in items_to_process if x.priority == 'low')} low priority items)"
        )

    counters: dict[str, int] = {}
    out: list[FeedbackStoredModel] = []
    for it in items_to_process:
        sec_slug = slug(it.section)
        counters[sec_slug] = counters.get(sec_slug, 0) + 1
        idx = counters[sec_slug] - 1
        item_id = f"{sec_slug}_iter{iteration}_item{idx}"
        out.append(
            FeedbackStoredModel(
                id=item_id,
                section=it.section,
                type=it.type,
                issue=it.issue,
                suggestion=it.suggestion,
                priority=it.priority,
                quote=it.quote,
                paragraph_number=it.paragraph_number,
                location_hint=it.location_hint,
                status=FeedbackStatus.PENDING,
            )
        )
    return out
