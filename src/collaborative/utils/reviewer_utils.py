import logging
import re
from typing import Dict, List

from src.collaborative.utils.models import (
    FeedbackStatus,
    FeedbackStoredModel,
    ReviewerTaskValidationModel,
)
from src.utils.data.models import Article

logger = logging.getLogger(__name__)


def extract_citation_ids(text: str) -> List[str]:
    """Extract citation IDs from text containing <c cite="..."/> tags."""
    citation_pattern = r'<c cite="([^"]+)"/>'
    return re.findall(citation_pattern, text)


def build_reference_map(article: Article, references: List[str]) -> Dict[str, str]:
    reference_map = {}
    for section in article.sections:
        # check if references exist in section
        for ref in references:
            if ref in section and ref not in reference_map:
                reference_map[ref] = section
    return reference_map


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
    Handles both single section strings and lists of sections.
    in:  ReviewerTaskValidationModel (LLM items without id/status)
    out: List[FeedbackStoredModel] (ready to store, limited to max_items)
    """
    invalid_section_names = ("_overall", "N/A", "n/a", "NA", "na", "")

    # Filter out invalid section names
    valid_items = []
    invalid_count = 0
    for item in validation.items:
        # Handle both str and List[str]
        sections = item.section if isinstance(item.section, list) else [item.section]

        # Check if any section is invalid
        if any(sec in invalid_section_names for sec in sections):
            invalid_count += 1
            logger.warning(
                f"Skipping feedback item with invalid section name(s) '{item.section}': {item.issue[:50]}..."
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
        # For multi-section items, use first section for ID generation
        first_section = it.section[0] if isinstance(it.section, list) else it.section
        sec_slug = slug(first_section)
        counters[sec_slug] = counters.get(sec_slug, 0) + 1
        idx = counters[sec_slug] - 1
        item_id = f"{sec_slug}_iter{iteration}_item{idx}"
        out.append(
            FeedbackStoredModel(
                id=item_id,
                section=it.section,  # Keep as list or str
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
