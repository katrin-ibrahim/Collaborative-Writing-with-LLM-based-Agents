from enum import Enum

import logging
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing import Any, Dict, List, Literal, Optional, Union

# --- Configuration Values ---

MAX_HEADING_COUNT = 20
# ----------------------------


class SearchSummary(BaseModel):
    source_queries: List[str]
    search_id: str
    results: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ArticleOutlineValidationModel(BaseModel):
    """The title and hierarchical structure for the article with chunk selection."""

    headings: List[str] = Field(
        ...,
        min_length=3,
        max_length=MAX_HEADING_COUNT,
        description=f"A list of up to {MAX_HEADING_COUNT} main section headings as plain strings (not objects).",
    )
    chunk_map: Dict[str, List[str]] = Field(
        ...,
        description="Mapping of section heading to list of relevant chunk IDs for that section.",
    )

    @field_validator("headings", mode="before")
    @classmethod
    def normalize_headings(cls, v: Any) -> List[str]:
        """
        Normalize headings to plain strings.
        Handles cases where LLM returns [{"title": "..."}, ...] instead of ["...", ...].
        """
        # Accept several input shapes from the LLM and coerce to list of items:
        # - already a list: use as-is
        # - a single string: wrap into [string]
        # - a dict: could be either a wrapper like {"headings": [...]}, or a single heading object like {"title": "..."}
        if isinstance(v, str):
            v = [v]
        elif isinstance(v, dict):
            # Try to find common container keys that hold the actual list
            for candidate in (
                "headings",
                "items",
                "sections",
                "sections_list",
                "titles",
            ):
                if candidate in v and isinstance(v[candidate], list):
                    v = v[candidate]
                    break
                # If container key maps to a dict, treat its keys as headings
                if candidate in v and isinstance(v[candidate], dict):
                    v = list(v[candidate].keys())
                    break
            else:
                # Treat the dict as a single heading object
                v = [v]

        if not isinstance(v, list):
            raise ValueError("headings must be a list")

        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                # Extract from {"title": "..."} or similar structures
                if "title" in item:
                    normalized.append(str(item["title"]))
                elif "heading" in item:
                    normalized.append(str(item["heading"]))
                elif "name" in item:
                    normalized.append(str(item["name"]))
                # If the dict contains a nested list of headings under common keys, extend from it
                elif any(
                    k in item and isinstance(item[k], list)
                    for k in ("headings", "items", "sections", "titles")
                ):
                    for k in ("headings", "items", "sections", "titles"):
                        if k in item and isinstance(item[k], list):
                            for sub in item[k]:
                                if isinstance(sub, str):
                                    normalized.append(sub)
                                elif isinstance(sub, dict):
                                    if "title" in sub:
                                        normalized.append(str(sub["title"]))
                                    elif "heading" in sub:
                                        normalized.append(str(sub["heading"]))
                                    elif "name" in sub:
                                        normalized.append(str(sub["name"]))
                                    elif len(sub) == 1:
                                        normalized.append(str(next(iter(sub.values()))))
                                    else:
                                        # Fallback: use keys if multiple
                                        normalized.extend([str(x) for x in sub.keys()])
                                else:
                                    normalized.append(str(sub))
                            break
                else:
                    # If multiple keys given (e.g., {"Heading A": "...", "Heading B": "..."}),
                    # treat the keys as headings.
                    if len(item) > 1:
                        normalized.extend([str(k) for k in item.keys()])
                    elif len(item) == 1:
                        # Single-key dict: use the sole value
                        normalized.append(str(next(iter(item.values()))))
                    else:
                        raise ValueError(f"Cannot extract heading from dict: {item}")
            else:
                # Try to convert to string
                normalized.append(str(item))

        return normalized

    @field_validator("headings", mode="after")
    @classmethod
    def truncate_excess_headings(cls, v: List[str]) -> List[str]:
        """Truncate to MAX_HEADING_COUNT if LLM returns too many headings."""
        if len(v) > MAX_HEADING_COUNT:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"LLM returned {len(v)} headings, truncating to {MAX_HEADING_COUNT}"
            )
            return v[:MAX_HEADING_COUNT]
        return v


# endregion Writer Validation Models


# region Reviewer Feedback Models


# region Enums
class FeedbackType(str, Enum):
    """Categories of feedback for gap-based review."""

    # accuracy, content_expansion, structure, clarity, and style.
    ACCURACY = "accuracy"
    CONTENT_EXPANSION = "content_expansion"
    STRUCTURE = "structure"
    CLARITY = "clarity"
    STYLE = "style"


class FeedbackStatus(str, Enum):
    """Feedback lifecycle status."""

    PENDING = "pending"
    ADDRESSED = "addressed"
    WONT_FIX = "wont_fix"
    VERIFIED_ADDRESSED = "verified_addressed"


#  endregion Enums


class CriticalContradiction(BaseModel):
    """A factual contradiction found in the article."""

    section: str = Field(description="Section name where the issue appears")
    claim: str = Field(description="The problematic claim from the article")
    evidence: str = Field(
        description="Why it's wrong (what the ground truth/chunks actually say)"
    )


class MissingCriticalFact(BaseModel):
    """An important fact that is missing from the article."""

    section: str = Field(description="Section where this fact should be added")
    fact: str = Field(description="Description of what's missing")
    suggested_evidence: str = Field(
        description="Where this information comes from (infobox or chunks)"
    )


class FactCheckValidationModel(BaseModel):
    critical_contradictions: List[CriticalContradiction] = Field(default_factory=list)
    missing_critical_facts: List[MissingCriticalFact] = Field(default_factory=list)


class FeedbackValidationModel(BaseModel):
    """
    ONE item as emitted by the Reviewer LLM.
    No id, no status, no iteration — Python adds those.
    """

    section: Union[str, List[str]] = Field(
        description="Section name(s) this applies to. Can be a single section name or a list of section names for cross-cutting issues."
    )
    type: FeedbackType = Field(description="Feedback category.")
    issue: str = Field(description="What is wrong?")
    suggestion: str = Field(
        default="Review and address the issue described above.",
        description="How to fix it.",
    )
    priority: str = Field(
        pattern="^(high|medium|low)$", description="Importance: high|medium|low"
    )
    quote: Optional[str] = Field(
        default=None, description="Optional exact excerpt linked to the issue."
    )
    paragraph_number: Optional[int] = Field(
        default=None, description="Optional 1-indexed paragraph."
    )
    location_hint: Optional[str] = Field(
        default=None, description="Optional locator hint within the section."
    )

    @field_validator("paragraph_number", mode="before")
    @classmethod
    def coerce_paragraph_number(cls, v: Any) -> Optional[int]:
        """Convert empty strings to None for paragraph_number."""
        if v == "" or v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None
        return v


class ReviewerQueryHint(BaseModel):
    """
    Structured query hint for intelligent chunk filtering during revision.
    """

    query: str = Field(
        description="Wikipedia page title (exact when possible) or search query string."
    )
    intent: str = Field(
        description="Purpose of this query (e.g., 'fill_venue_details', 'add_player_stats', 'expand_background')."
    )
    expected_fields: List[str] = Field(
        default_factory=list,
        description="Short canonical field names or concepts to look for (e.g., ['stadium', 'capacity', 'location']).",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Optional fallback matching tokens or phrases for filtering chunks.",
    )
    note: Optional[str] = Field(
        default=None,
        description="Optional human-readable explanation of why this query is needed.",
    )


class ReviewerTaskValidationModel(BaseModel):
    """
    Reviewer LLM output: section-level feedback items and suggested search queries.
    """

    items: List[FeedbackValidationModel] = Field(
        description="A list of specific feedback items, one for each issue found. Must not be empty."
    )
    suggested_queries: List[ReviewerQueryHint] = Field(
        default_factory=list,
        description="Structured search query hints with intent and filtering keywords for targeted chunk retrieval.",
    )


# --------- Stored item (single source of truth) ----------
class FeedbackStoredModel(BaseModel):
    """
    Canonical stored feedback item. Mirrors validation fields, plus id/status/writer_comment.
    Iteration is NOT a field — it's the container key in memory: feedback_by_iteration[iteration][id] = item.
    """

    model_config = ConfigDict(extra="forbid")

    # from validation
    section: Union[str, List[str]] = Field(
        description="Section name(s) this applies to. Can be a single section name or a list of section names."
    )
    type: FeedbackType = Field(description="Feedback category.")
    issue: str = Field(description="What is wrong?")
    suggestion: str = Field(
        default="Review and address the issue described above.",
        description="How to fix it.",
    )
    priority: str = Field(
        pattern="^(high|medium|low)$", description="Importance: high|medium|low"
    )
    quote: Optional[str] = Field(default=None, description="Optional exact excerpt.")
    paragraph_number: Optional[int] = Field(
        default=None, description="Optional 1-indexed paragraph."
    )
    location_hint: Optional[str] = Field(
        default=None, description="Optional locator hint within the section."
    )

    @field_validator("paragraph_number", mode="before")
    @classmethod
    def coerce_paragraph_number(cls, v: Any) -> Optional[int]:
        """Convert empty strings to None for paragraph_number."""
        if v == "" or v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None
        return v

    # added by Python
    id: str = Field(description="Globally unique id (e.g., 'intro_iter2_item0').")
    status: FeedbackStatus = Field(
        default=FeedbackStatus.PENDING, description="Current resolution status."
    )
    writer_comment: Optional[str] = Field(
        default=None, description="Writer's note when updating status."
    )


# --------- Writer → validation ----------
class WriterStatusUpdate(BaseModel):
    id: str = Field(description="Existing feedback item id to update.")
    status: FeedbackStatus = Field(description="One of: pending | addressed | wont_fix")
    writer_comment: Optional[str] = Field(
        default=None, description="Short rationale/note."
    )
    # NOTE: if status == VERIFIED_ADDRESSED is returned, we will ignore it (writer can't verify).


class WriterValidationModel(BaseModel):
    """Model for writer's feedback status updates."""

    section_name: str = Field(description="The section name this update applies to.")
    updates: List[WriterStatusUpdate] = Field(
        description="Per-item status updates from the writer."
    )
    updated_content: str = Field(
        description="The revised section content after addressing feedback."
    )


class WriterValidationBatchModel(BaseModel):
    """Batch model for multiple writer feedback status updates."""

    items: List[WriterValidationModel] = Field(
        description="List of writer validation models for batch processing."
    )


class ResearchGateDecision(BaseModel):
    action: Literal["continue", "retry"] = Field(
        description="continue to outline, or retry to do more research"
    )
    delete_chunk_ids: List[str] = Field(
        default_factory=list, description="Chunk IDs to delete"
    )
    new_queries: List[str] = Field(
        default_factory=list,
        description="New Wikipedia page titles to search (only if action=retry)",
    )
    reasoning: str = Field(default="", description="Brief explanation of decision")


# --------- Verifier → validation ----------
class VerifierStatusUpdate(BaseModel):
    id: str = Field(description="Existing feedback item id.")
    status: FeedbackStatus = Field(
        description="Use 'verified_addressed' to confirm; 'pending' to revert; may keep 'wont_fix' if justified."
    )


class VerificationValidationModel(BaseModel):
    updates: List[VerifierStatusUpdate] = Field(
        description="Per-item verification decisions."
    )

    @model_validator(mode="before")
    @classmethod
    def wrap_llm_shortcut(cls, data: Any) -> Any:
        """
        Catches LLM outputs that are just the dict of updates,
        not the full object. Also handles LLM returning "items" instead of "updates".
        """
        if isinstance(data, dict):
            # LLM might return {"items": [...]} instead of {"updates": [...]}
            if "items" in data and "updates" not in data:
                data = {"updates": data["items"]}
            elif "updates" not in data:
                # The LLM returned the raw dictionary, e.g., {"id-1": "pending", ...}
                # Wrap it in the expected structure so validation can proceed.
                return {"updates": data}
        return data

    @field_validator("updates", mode="before")
    @classmethod
    def normalize_updates(cls, v: Any) -> List[dict]:
        """
        Normalize updates to list of dicts.
        Handles cases where LLM returns {"id=...": "status", ...} instead of [{"id": "...", "status": "..."}, ...].
        """
        # Handle extra list wrapping: [[{...}]] -> [{...}]
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
            v = v[0]

        if isinstance(v, list):
            # Already in the correct list format, e.g., [{"id": "...", "status": "..."}]
            # But check if any items are incorrectly wrapped
            normalized = []
            for item in v:
                if isinstance(item, dict):
                    # Check if status field is incorrectly a list
                    if "status" in item and isinstance(item["status"], list):
                        # If it's an empty list, skip this item entirely
                        if len(item["status"]) == 0:
                            continue
                        # If it's a list with one element, unwrap it
                        elif len(item["status"]) == 1:
                            item["status"] = item["status"][0]
                        else:
                            # Multiple elements - take the first one
                            item["status"] = item["status"][0]
                    normalized.append(item)
                elif (
                    isinstance(item, list)
                    and len(item) == 1
                    and isinstance(item[0], dict)
                ):
                    # Unwrap single-item list
                    unwrapped = item[0]
                    # Check status field in unwrapped item too
                    if "status" in unwrapped and isinstance(unwrapped["status"], list):
                        if len(unwrapped["status"]) == 0:
                            continue
                        elif len(unwrapped["status"]) == 1:
                            unwrapped["status"] = unwrapped["status"][0]
                        else:
                            unwrapped["status"] = unwrapped["status"][0]
                    normalized.append(unwrapped)
                elif isinstance(item, list):
                    # Skip empty lists or malformed items
                    continue
                else:
                    normalized.append(item)
            return normalized
        elif isinstance(v, dict):
            # LLM returned a dict mapping id→status (e.g., {"id=xyz": "pending"})
            # Convert to list of VerifierStatusUpdate-compatible dicts
            normalized = []
            for k, status_val in v.items():
                # Extract the id: strip "id=" prefix if present
                id_str = k.replace("id=", "").strip()
                normalized.append({"id": id_str, "status": status_val})
            return normalized
        else:
            raise ValueError("updates must be a list or dict")


class SectionContentModel(BaseModel):
    """Model for a single section's content."""

    section_heading: str = Field(description="The section heading being written.")
    content: str = Field(description="The written content for this section.")
    summary: str = Field(
        description="A brief 1-2 sentence summary of what this section covers, for maintaining coherence in subsequent sections."
    )


class BatchSectionWritingModel(BaseModel):
    """Model for batch writing multiple sections at once."""

    sections: List[SectionContentModel] = Field(
        description="List of section content models for batch writing."
    )

    @field_validator("sections", mode="before")
    @classmethod
    def filter_malformed_sections(cls, v: Any) -> List[dict]:
        """
        Filter out malformed section items that are strings or incomplete objects.
        """
        if not isinstance(v, list):
            return v

        cleaned = []
        for item in v:
            # Skip non-dict items (malformed JSON fragments like '{' or incomplete strings)
            if not isinstance(item, dict):
                continue
            # Skip items missing required fields
            if "section_heading" not in item or "content" not in item:
                continue
            cleaned.append(item)

        return cleaned


# endregion Reviewer Feedback Models
