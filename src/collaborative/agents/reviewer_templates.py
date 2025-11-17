from typing import Any, Dict, List, Optional

from src.utils.data import Article


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

# ---------------------------------------- Unified Factual Verification Template ---------------------------------------


def build_fact_check_prompt_v2(
    article: Article,
    infobox_data: str,
    cited_chunks: str,
    ref_map: Dict[str, str],
) -> str:
    """
    Unified fact checking prompt that verifies article claims against both infobox and cited chunks.

    Args:
        article: The article to fact-check
        infobox_data: Formatted infobox facts (ground truth)
        cited_chunks: Content of all cited chunks
        ref_map: Mapping of citations to chunk IDs
    """
    return f"""
You are a fact-checker verifying an article about "{article.title}".

ARTICLE CONTENT:
{article.content}

GROUND TRUTH INFOBOX FACTS:
{infobox_data if infobox_data else "No infobox data available."}

CITED RESEARCH CHUNKS:
{cited_chunks if cited_chunks else "No citations found."}

CITATION MAP (which chunks are cited where):
{ref_map if ref_map else "No citations mapped."}

TASK: Identify factual issues in TWO categories:

1. CRITICAL CONTRADICTIONS:
   - Claims that directly contradict the infobox ground truth
   - Claims not supported by the cited chunks
   - Incorrect facts (dates, names, scores, locations, etc.)

   For each contradiction, provide:
   - section: Section name where the issue appears
   - claim: The problematic claim from the article
   - evidence: Why it's wrong (what the ground truth/chunks actually say)

2. MISSING CRITICAL FACTS:
   - Important facts from the infobox that should be in the article but aren't
   - Key information that would improve factual completeness

   For each missing fact, provide:
   - section: Section where this fact should, if it is not in a single specific section you may suggest multiple sections
   - fact: Description of what's missing
   - suggested_evidence: Where this information comes from (infobox or chunks)

GUIDELINES:
- Be strict about factual accuracy - any deviation is a contradiction
- Focus on the most important facts first
- If ground truth and article match, no contradiction
- Only flag truly missing critical facts, not minor details
- Limit to top 5 most critical issues per category

Return a JSON object matching the FactCheckValidationModel schema:
{{
  "critical_contradictions": [
    {{"section": "...", "claim": "...", "evidence": "..."}}
  ],
  "missing_critical_facts": [
    {{"section": "...", "fact": "...", "suggested_evidence": "..."}}
  ]
}}
"""


def build_review_prompt_v2(
    article: Article,
    fact_check_results: str,
    chunk_summaries: str,
    max_suggested_queries: int,
    possible_searches: List[str],
    tom_context: Optional[str] = None,
) -> str:
    """
    Generate review feedback prompt with fact-check results and research context.

    Args:
        article: The article being reviewed
        fact_check_results: Formatted fact-check results from unified verification
        chunk_summaries: Available research chunks
        max_suggested_queries: Maximum number of search queries to suggest
        possible_searches: Potential new searches from category extraction
        tom_context: Optional theory of mind context
    """
    tom_section = ""
    if tom_context:
        tom_section = f"""
STRATEGIC CONTEXT (Theory of Mind):
{tom_context}

STRATEGIC GUIDANCE FOR REVIEWER:
Based on the predicted writer behavior, adjust your feedback strategy:
- If writer likely to accept most feedback: Provide comprehensive, detailed feedback covering multiple dimensions
- If writer likely to partially accept: Prioritize high-impact items; focus on critical accuracy and missing facts first
- If writer likely to contest: Be extra specific with evidence; reduce volume to highest-priority issues only

Calibrate feedback volume and specificity to maximize acceptance rate and collaboration efficiency.
"""

    possible_searches_str = (
        "\n".join(f"- {s}" for s in possible_searches[:15])
        if possible_searches
        else "None available"
    )

    return f"""
You are a world-class editor reviewing an article about "{article.title}".

ARTICLE CONTENT:
{article.content}

FACT-CHECK RESULTS:
{fact_check_results}

AVAILABLE RESEARCH CHUNKS (for suggesting improvements or expansions):
{chunk_summaries}

POTENTIAL NEW SEARCHES (from category analysis):
{possible_searches_str}

{tom_section}

YOUR TASK:
1. Review the fact-check results - address all critical contradictions and missing facts first
2. Provide constructive feedback on content quality (structure, clarity, completeness)
3. Suggest specific research chunks that could improve sections
4. Suggest up to {max_suggested_queries} new Wikipedia searches with structured hints for filtering

FEEDBACK GUIDELINES:
- Priority 1: Factual correctness (address all contradictions first)
- Priority 2: Missing critical information
- Priority 3: Structure, clarity, and encyclopedic quality
- Be specific about which section needs improvement
- Reference specific chunk IDs when suggesting research
- Keep feedback actionable and constructive
- Provide 3-5 high-impact feedback items per iteration (focus on quality over quantity)

CRITICAL: SECTION ASSIGNMENT RULES:
- For existing content issues: Assign to the specific section name where the problem exists
- For new content/missing information: Assign to the section where you think it should go
- For issues affecting multiple sections: Use a JSON array: ["Section1", "Section2"]
- DO NOT OUTPUT FEEDBACK WITHOUT A VALID SECTION NAME(S)

EXAMPLES:
- "Introduction is missing the date" → section: "Introduction"
- "Add information about X suggestion: "Add to section Y and Z with information about X from chunk_123"
- "Sections Y and Z have inconsistent tone" → section: ["Y", "Z"]

STRUCTURED QUERY HINTS:
When suggesting searches, provide structured hints to enable intelligent filtering:
- query: Exactly as seen in POTENTIAL NEW SEARCHES
- intent: Short purpose description (e.g., "fill_venue_details", "add_player_stats")
- expected_fields: List of canonical concepts to look for (e.g., ["capacity", "location", "attendance"])
- keywords: Optional fallback terms for matching (e.g., ["stadium", "venue", "crisis"])
- note: Optional explanation of why this search is needed

OUTPUT FORMAT:
Return a JSON object matching ReviewerTaskValidationModel schema:
{{
  "items": [
    {{
      "section": "section name",
      "type": "accuracy|content_expansion|structure|clarity|style",
      "issue": "description of the problem",
      "suggestion": "specific actionable suggestion",
      "priority": "high|medium|low",
      "quote": "optional exact quote from article",
      "paragraph_number": 1,
      "location_hint": "optional location hint"
    }}
  ],
  "suggested_queries": [ // max {max_suggested_queries}
    {{
      "query": "Wikipedia page title (from potential searches only)",
      "intent": "purpose_description",
      "expected_fields": ["field1", "field2"],
      "keywords": ["keyword1", "keyword2"],
      "note": "optional explanation"
    }}
  ]
}}

Focus on quality over quantity - provide thorough feedback on the most important issues.
"""


def build_reviewer_tom_prediction_prompt(
    interaction_history: List[Dict[str, Any]],
    feedback_context: dict,
) -> str:
    """
    Build Theory of Mind prediction prompt for Reviewer predicting Writer's response.

    Args:
        interaction_history: List of all past observed agent actions
        feedback_context: Dict with feedback_count, feedback_types, iteration

    Returns:
        Prompt for reviewer to predict writer's likely response to feedback
    """
    feedback_count = feedback_context.get("feedback_count", "unknown")
    iteration = feedback_context.get("iteration", 0)

    writer_actions = [
        "accept_most_feedback",
        "partially_accept_feedback",
        "contest_some_feedback",
    ]

    history_lines = []
    if interaction_history:
        for obs in interaction_history:
            history_lines.append(
                f"- Iteration {obs.get('iteration')}: {obs.get('agent')} observed doing '{obs.get('action')}'"
            )
    history_log = "\n".join(history_lines) if history_lines else "No history yet."

    return f"""You are a REVIEWER agent predicting the WRITER's response to your feedback.

CURRENT SITUATION:
- You are about to provide feedback for iteration {iteration}.
- Feedback items being provided: {feedback_count}

INTERACTION HISTORY:
{history_log}

PREDICTION TASK:
Analyze the sequence of past interactions to identify patterns in the Writer's behavior.
Based on this history and the feedback you are *about to provide*, predict how the writer will respond.

AVAILABLE WRITER ACTIONS:
{writer_actions}

PREDICTION GUIDELINES:
- accept_most_feedback: Writer typically addresses 70%+ of feedback
- partially_accept_feedback: Writer addresses 40-70% of feedback selectively
- contest_some_feedback: Writer addresses <40% of feedback

Consider:
- The Writer's historical acceptance patterns (e.g., "accept_most_feedback").
- The volume and nature of your *current* feedback (e.g., {feedback_count} items).
- The iteration stage (later iterations might see more 'contest').

OUTPUT:
- predicted_action: ONE of the actions from the list above
- confidence: 0.0 to 1.0 based on how certain you are
- reasoning: 2-3 sentences explaining your prediction based on the history.

Your prediction will help you calibrate the tone and volume of your feedback."""
