from typing import Dict, List, Optional

from src.utils.data import Article


def build_review_prompt(
    article: Article,
    validation_results: Dict,
    tom_context: Optional[str],
    chunk_summaries: Optional[str] = None,
    max_suggested_queries: int = 5,
    infobox_data: Optional[Dict] = None,
    related_articles: Optional[List[str]] = None,
) -> str:
    """Builds the single, holistic review prompt for the agent."""

    # Format citation validation results
    validation_text = f"""
Citation Validation (for `accuracy` checking):
- Total citations: {validation_results.get('total_citations', 0)}
- Valid citations: {validation_results.get('valid_citations', 0)}
- Missing chunks: {len(validation_results.get('missing_chunks', []))}
- Needs source tags: {validation_results.get('needs_source_count', 0)}
"""

    infobox_context_str = ""
    if infobox_data:
        infobox_context_str += (
            "TOPIC INFOBOX: The following are key facts from the article's Infobox:\n"
        )
        infobox_items = [
            f"- {key}: {value}" for key, value in infobox_data.items() if value
        ]
        if infobox_items:
            infobox_context_str = "\n".join(infobox_items)

    related_articles_str = "No related articles found."
    if related_articles:
        related_articles_str = "\n".join([f"- {name}" for name in related_articles])

    # Include research context if provided (now chunk summaries)
    research_section = ""
    if chunk_summaries:
        research_section = f"""
        ## CHUNK SUMMARIES AND VALIDATION RESULTS
        {validation_text}
        {chunk_summaries}
        """
    else:
        research_section = f"""
        ## CHUNK SUMMARIES
        {validation_text}
        No research chunk summaries found in memory.
        """

    guidelines = f"""
You are a world-class editor reviewing an article on "{article.title}".
Your goal is to provide holistic, actionable feedback. You must check for **Relevance (Gaps)** and **Coherence (Style)** at the same time.

**REVIEW TASKS (Do all as needed):**

1.  **Check for Factual Gaps & Relevance :**
    * **Compare to `Infobox`:** If a simple fact is missing (e.g., a winner, date), create a `content_expansion` item. In the `suggestion`, write the *exact fact* to add. In the `section` field, name the section where it belongs (e.g., "Event Details").
    * **Compare to `Chunk Summaries`:** If the writer missed a detail that's already in the research, create a `content_expansion` or `clarity` item. In the `suggestion`, list the *exact `chunk_ids`* to re-examine. In the `section` field, name the section where it belongs.
    * **Compare to `Potential New Research Paths`:** If a section is lacking depth, **decide** if a new search using only up to {max_suggested_queries} from {related_articles_str} can fix it. If yes, add *one* of the related article names (e.g., "History of the Venue") to the `suggested_queries` list.

2.  **Check for Accuracy & Verifiability:**
    * Review the `Citation Validation` report above.
    * If a claim is factually wrong, unverified, or contradicts a source, create an `accuracy` item.

3.  **Check for Coherence & Style:**
    * Review the article's flow, organization, and tone.
    * Create `structure` items for confusing order or bad flow.
    * Create `clarity` items for confusing writing.
    * Create `style` items for repetitive text or non-encyclopedic tone.

Use any of the 5 allowed feedback types as needed to cover all issues.
The section will be indicated by H2 headings in the article.
"""

    # ToM context if available
    tom_section = ""
    if tom_context:
        tom_section = f"\n\nCollaborative Context (Theory of Mind):\n{tom_context}\n"

    return f"""You are reviewing an article an article about "{article.title}".

ARTICLE TO REVIEW:
{article.content}

REVIEW GUIDELINES:
{guidelines}
{infobox_context_str}
{research_section}

{tom_section}

REQUIRED OUTPUT FORMAT:
Your response must be a valid JSON object with the following structure:
{{
  "items": [
    {{
      "section": "The EXACT name of the section this feedback applies to. MUST NOT be '_overall'.",
      "type": "MUST be exactly one of these 5 values: accuracy, content_expansion, structure, clarity, style",
      "issue": "clear description of what is wrong",
      "suggestion": "specific actionable recommendation to fix it",
      "priority": "high, medium, or low",
      "quote": "optional: exact text excerpt related to the issue",
      "paragraph_number": "optional: 1-indexed paragraph number",
      "location_hint": "optional: additional location context"
    }}
  ],
  "suggested_queries": []
}}

CRITICAL RULES:
- The "type" field MUST be EXACTLY one of these 5 strings:
1. accuracy
2. content_expansion
3. structure
4. clarity
5. style
- The "section" field MUST be one of the section names from the list above.".
- If you are suggesting new content (`content_expansion`), set the "section" field to the name of the section where this new content should be placed.

GUIDELINES:
- Create one item per distinct issue found
- Assign priority based on impact: high (critical), medium (important), low (minor)
- Use the 'quote' field when referencing specific text
- Be specific and actionable in your suggestions
"""


def build_verification_prompt(
    previous_iteration: int,
    addressed_items: List[str],
    pending_items: List[str],
    wont_fix_items: List[str],
    needs_clarification_items: List[str],
    article: Article,
    current_iteration: int,
) -> str:
    """Build verification prompt with categorized feedback."""
    return f"""You are verifying how the Writer responded to feedback from Iteration {previous_iteration}.

CURRENT ARTICLE (Iteration {current_iteration}):
Title: {article.title}

Content:
{article.content}

---

FEEDBACK STATUS FROM ITERATION {previous_iteration}:

ITEMS MARKED AS ADDRESSED ({len(addressed_items)}):
{chr(10).join(addressed_items) if addressed_items else '  (None)'}

ITEMS NOT YET ADDRESSED ({len(pending_items)}):
{chr(10).join(pending_items) if pending_items else '  (None)'}

ITEMS MARKED AS WON'T FIX ({len(wont_fix_items)}):
{chr(10).join(wont_fix_items) if wont_fix_items else '  (None)'}

ITEMS NEEDING CLARIFICATION ({len(needs_clarification_items)}):
{chr(10).join(needs_clarification_items) if needs_clarification_items else '  (None)'}

---

VERIFICATION TASK:
Analyze the current article and evaluate each listed feedback item. For EVERY item shown above, decide one status:
- "verified_addressed": the writer's change is present and correct
- "pending": still not fixed or insufficiently addressed
- "wont_fix": you agree the item should remain won't-fix given the writer's rationale

IMPORTANT ID RULE:
- Each line includes an id token in the form "id=<ID>" when available. Use that exact ID string in your output.
- If a line does not show an id, omit it from updates (do NOT invent ids).


GUIDELINES:
- Include an update for EVERY item that shows an id=...
- Base decisions ONLY on the current article content and the item description
- Keep the summary concise and factual

CRITICAL: USE ONLY THE KEY 'updates' IN YOUR OUTPUT JSON, WHICH MUST BE A LIST OF OBJECTS WITH 'id' AND 'status' KEYS.
"""


# endregion Reviewer Prompt Templates

# ---------------------------------------- Claim Verification Template ---------------------------------------
CLAIM_VERIFICATION_PROMPT = """You are verifying whether a claim is supported by the provided source chunks.

CLAIM:
{claim}

SOURCE CHUNKS:
{chunks}

Task: Determine if the claim is SUPPORTED or NOT SUPPORTED by the source chunks.

Rules:
- The claim is SUPPORTED if the chunks provide direct evidence for it
- The claim is NOT SUPPORTED if:
  * The chunks don't contain information about the claim
  * The chunks contradict the claim
  * The chunks provide only tangential or unrelated information

Respond with one of:
- "SUPPORTED: The chunks provide evidence for this claim."
- "NOT SUPPORTED: [brief reason why chunks don't support claim]"

Keep your response concise (1-2 sentences).
"""
