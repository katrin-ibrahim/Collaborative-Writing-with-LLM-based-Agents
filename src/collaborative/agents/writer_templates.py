from typing import Any, Dict, List, Optional

from src.collaborative.utils.models import FeedbackStoredModel
from src.utils.data import Article

# ---------------------------------------- Writer Templates ----------------------------------------
# region Writer Prompt Templates
HEADING_COUNT = 15


def build_outline_prompt(topic: str, chunk_summaries: str) -> str:
    """
    Generates the prompt for creating a structured article outline using LLM.
    Also asks LLM to select relevant chunks for each section

    Args:
        topic: The main subject of the article.
        chunk_summaries: The formatted summaries of all stored research chunks (with chunk_ids).

    Returns:
        The prompt string instructing the LLM to output outline + chunk selections.
    """
    return f"""
    Generate a Wikipedia-style article outline for the topic: "{topic}" AND select relevant research chunks for each section.

    CRITICAL TOPIC FOCUS:
    - This article MUST be specifically about "{topic}" - not related topics, background events, or similar subjects.
    - TEMPORAL CONSTRAINT: If the topic contains a specific year/date, ALL sections must be about that EXACT time period.
      * REJECT any sections mentioning different years, different seasons, or different time periods.
      * Example: For a 2022 event, REJECT sections about 2018, 2023, or any other year.
    - ENTITY CONSTRAINT: If the topic is about a specific event, DO NOT create sections about related entities' broader histories.
      * Example: For "2022 Grand Final", REJECT sections about team season performance, venue history, or championship history.
    - If the research context mentions related but different topics (e.g., different competitions, different years, different categories),
      DO NOT create sections about those topics. Focus ONLY on "{topic}" itself.
    - Every section heading must directly relate to "{topic}" - reject any tangential or background topics that distract from the main subject.
    - When in doubt: "Is this section specifically about '{topic}'?" If no, exclude it.

    Use the research context to identify key entities such as people, organizations, events, locations, and outcomes.
    Headings should reflect these specific entities and capture factual, verifiable aspects of the topic.

    RESEARCH CONTEXT (with chunk_ids):
    ---
    {chunk_summaries}
    ---


    Guidelines for Wikipedia-style section headings:
    - Each heading must refer to a concrete entity, event, or concept found in the context.
    - Prioritize entity-rich headings that include NAMES of people, places, organizations, or specific events.
      Examples: "Response by Pakistani Government", "Impact in Sindh Province", "International Aid from UN"
      (NOT: "Government Response", "Regional Impact", "International Aid")

    - CRITICAL: DO NOT fabricate or guess specific entity names (venues, people, organizations) not explicitly mentioned in the research context
    - If the research context does not mention a specific venue name, use generic headings like "Venue and Location" (NOT "Match Location: [Guessed Stadium Name]")
    - If the research context does not mention specific people, use generic headings like "Key Players" (NOT "Performance by [Guessed Player Name]")
    - Only include specific entity names in headings when those entities are EXPLICITLY mentioned in the provided research chunks

    - For events about disasters/incidents: prefer sections like "Background", "Impact in [Region]", "Response", "Aftermath"
    - For events about elections: prefer sections like "Background", "Candidates", "Campaign", "Results", "Aftermath"
    - For events about conflicts: prefer sections like "Background", "Belligerents", "Course of battle", "Casualties", "Aftermath"

    - Maintain a logical flow from background and participants to event details, consequences, and legacy.
    - Avoid abstract or analytical titles (e.g., "Analysis of Causes", "Climate Factors", "Implications").
    - Avoid generic process-oriented titles (e.g., "How it Happened", "Why it Occurred").
    - Use short noun phrases naming the specific subject (ideally under 8 words).
    - Keep the tone factual, neutral, and encyclopedic — matching real Wikipedia article structure.
    - Ensure exactly {HEADING_COUNT} headings in total.
    - Headings must be a flat list of strings — no objects, no key–value pairs, no nested arrays, no annotations.

    CHUNK SELECTION TASK:
    For EACH section heading you create, select the most relevant chunk IDs from the research context above.
    - Select at least 1-3 chunks per section (use chunk_id field from research context)
    - Prioritize chunks that directly discuss the section's topic
    - Use ONLY the literal chunk IDs provided in the research context
    - DO NOT invent, shorten, or modify chunk IDs
    - If no perfect matches exist, select the most INDIRECTLY relevant chunks


    OUTPUT FORMAT:
    Return a JSON object with TWO fields:
    {{
      "headings": ["Background", "History", "Match Summary", ...],
      "chunk_map": {{
        "Background": ["chunk_id_1", "chunk_id_2"],
        "History": ["chunk_id_3", "chunk_id_5"],
        "Match Summary": ["chunk_id_4", "chunk_id_6"],
        ...
      }}
    }}

    CRITICAL: Every heading in "headings" must have a corresponding entry in "chunk_map" with at least 1 chunk ID.
    """


# Core reusable components
CORE_WRITING_RULES = """
WRITING STYLE - ENGAGING ENCYCLOPEDIA:
- Lead with concrete, specific details rather than generic statements
- Use precise numbers, dates, names, and facts from the research
- Avoid clichés like "highly anticipated", "storied", "remarkable", "significant"
- When describing events, focus on what actually happened rather than abstract qualities
- Build narrative through factual progression (what happened first, then what, then what)
- Let specific details create interest rather than relying on evaluative adjectives
- Include context that helps readers understand why events matter

REQUIREMENTS:
- Write AT LEAST 2-3 substantive, detailed paragraphs of encyclopedia-style content per section
- Each paragraph should be 4-6 sentences with rich factual detail
- Ground ALL statements in the provided research information
- Include as many specific details and examples from the sources as possible
- Maintain an informative, objective tone while being comprehensive
- Ensure smooth transitions and logical flow
- Start directly with the content, no preamble or introduction
- NEVER write placeholder content like "Information is unavailable" - use ALL relevant research provided
"""

CORE_CITATION_RULES = """
CITATION RULES - CRITICAL:
• Research chunks start with their ID (e.g., "abc123 {content: ...}")
• Cite using EXACT chunk ID in tags: <c cite="abc123"/>
• NEVER invent chunk IDs not in the research above
• NEVER use placeholders like [1], chunk_id_1, source_1, etc.
• If no research provided, write WITHOUT citations
• Example: "The event occurred in 2022 <c cite="947265bb2a2267de817bdef27b97a6dd"/>."
"""

CORE_TOPIC_FOCUS = """
CRITICAL TOPIC FOCUS:
- This content MUST be about "{topic}" specifically - NOT related topics, background events, or similar subjects.
- If the research mentions related but different topics (e.g., different leagues, years, categories), extract ONLY information that directly relates to "{topic}".
- When in doubt: Does this fact apply to "{topic}" specifically? If no, exclude it.
"""

CORE_NO_METACOMMENTARY = """
CRITICAL: Write ONLY the article content. Do NOT include:
- Conversational language ("Okay, here's a section...")
- Questions to the user ("What is the overall article's focus?")
- Meta-commentary about the writing process
- Placeholder text or requests for clarification
"""


def build_write_section_prompt(
    section_heading: str,
    topic: str,
    relevant_info: str,
    previous_summaries: Optional[List[dict]] = None,
) -> str:
    """
    Simple section writing prompt for fallback mode (when batch writing fails).

    Args:
        section_heading: Section heading to write
        topic: Article topic
        relevant_info: Research chunks (formatted with IDs)
        previous_summaries: Optional list of dicts with 'section_heading' and 'summary' from already-written sections
    """
    research_block = (
        f"RESEARCH INFORMATION:\n{relevant_info}"
        if relevant_info
        else "RESEARCH INFORMATION: None provided."
    )

    # Build previous context section if available
    context_section = ""
    if previous_summaries:
        context_items = [
            f"- **{item['section_heading']}**: {item['summary']}"
            for item in previous_summaries
        ]
        context_section = f"""
PREVIOUSLY WRITTEN SECTIONS (for coherence):
{chr(10).join(context_items)}

IMPORTANT: Build on this context naturally. Avoid repeating information already covered.
Reference earlier sections where appropriate (e.g., "As mentioned in the Background...").

"""

    return f"""
Write a comprehensive section titled "{section_heading}" for an article about "{topic}".

{context_section}{CORE_TOPIC_FOCUS.format(topic=topic)}
{CORE_NO_METACOMMENTARY}
{research_block}
{CORE_WRITING_RULES}
{CORE_CITATION_RULES}

REQUIREMENTS:
- Write AT LEAST 300 words (approximately 2-4 substantive, detailed paragraphs)
- Write in a vivid, engaging, and comprehensive style that captures reader interest
- Use narrative techniques where appropriate: vivid descriptions, compelling details, smooth transitions
- Extract and incorporate ALL relevant facts, details, names, dates, and statistics from the research

OUTPUT FORMAT (JSON):
{{
  "section_heading": "{section_heading}",
  "content": "Multiple detailed paragraphs with comprehensive coverage of the research provided",
  "summary": "Brief 1-2 sentence summary of this section's main points"
}}
"""


def build_write_sections_batch_prompt(
    sections_with_chunks: List[dict],
    topic: str,
    previous_summaries: Optional[List[dict]] = None,
) -> str:
    """
    Generate prompt for batch writing multiple sections at once.

    Args:
        sections_with_chunks: List of dicts with 'section_heading' and 'relevant_info' (chunks)
        topic: Article topic
        previous_summaries: Optional list of dicts with 'section_heading' and 'summary' from already-written sections
    """
    # Build previous context section if available
    context_section = ""
    if previous_summaries:
        context_items = [
            f"- **{item['section_heading']}**: {item['summary']}"
            for item in previous_summaries
        ]
        context_section = f"""
PREVIOUSLY WRITTEN SECTIONS (for coherence):
{chr(10).join(context_items)}

IMPORTANT: Build on this context naturally. Avoid repeating information already covered.
Reference earlier sections where appropriate (e.g., "As mentioned in the Background...").
"""

    sections_info = []
    for i, item in enumerate(sections_with_chunks, 1):
        section_heading = item["section_heading"]
        relevant_info = item.get("relevant_info", "")
        sections_info.append(
            f"""=== Section {i}: {section_heading} ===
Research chunks: {relevant_info if relevant_info else "None provided."}"""
        )

    sections_text = "\n\n".join(sections_info)

    return f"""
Write encyclopedia-style content for {len(sections_with_chunks)} sections of an article about "{topic}".

{context_section}

{sections_text}

{CORE_TOPIC_FOCUS.format(topic=topic)}
{CORE_NO_METACOMMENTARY}
{CORE_WRITING_RULES}
{CORE_CITATION_RULES}

BATCH OUTPUT REQUIREMENTS:
- For EACH section above, write AT LEAST 300 words (approximately 2-4 substantive, detailed paragraphs)
- Write in a vivid, engaging, and comprehensive style that captures reader interest (not just dry recitation of facts)
- Use narrative techniques where appropriate: descriptions, compelling details, smooth transitions between ideas, while still maintaining encyclopedic tone
- Extract and incorporate ALL relevant facts, details, names, dates, and statistics from the research chunks
- Use ONLY the research chunks provided for that specific section
- Each section should be comprehensive and information-rich, not brief summaries
- Avoid repetition between sections
- Generate a brief 1-2 sentence summary of what each section covers (for subsequent batches)
- NEVER write placeholder content or state that information is unavailable - fully utilize provided research

OUTPUT FORMAT (JSON), OUTPUT EXACTLY, make sure to include ALL KEYS:
{{
  "sections": [
    {{
      "section_heading": "exact heading from above",
      "content": "Multiple detailed paragraphs with comprehensive coverage of the research provided",
      "summary": "Brief 1-2 sentence summary of this section's main points"
    }}
  ]
}}

Write detailed, comprehensive content for ALL {len(sections_with_chunks)} sections listed above.
"""


def safe_trim(text: str, max_chars: int = 6000) -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n...\n" + text[-tail:]


def fmt_feedback_line(it: "FeedbackStoredModel") -> str:
    parts = [
        f"id={it.id}",
        f"type={it.type}",
        f"priority={it.priority}",
        f"issue={it.issue}",
        f"suggestion={it.suggestion}",
    ]
    if it.quote:
        parts.append(f"quote={it.quote}")
    if it.paragraph_number:
        parts.append(f"paragraph={it.paragraph_number}")
    if it.location_hint:
        parts.append(f"loc={it.location_hint}")
    return " | ".join(parts)


def build_research_gate_prompt(
    topic: str,
    chunks_formatted: str,
    wiki_links: Optional[str] = None,
    tried_queries: str = "None yet",
    retry_count: int = 0,
) -> str:
    """Pre-outline research filter with retry capability and clear grounding rules."""

    links_ctx = ""
    if wiki_links and wiki_links.strip().lower() != "none":
        links_ctx = f"""
Pre-fetched Wikipedia links (already reviewed):
{wiki_links}
You may only select or rephrase entities from these links or the chunk summaries below.
Do NOT invent entities outside this information.
"""

    retry_instruction = ""
    if retry_count < 3:
        retry_instruction = f"""
Retry logic:
- Current attempt: {retry_count + 1}/3
- Already tried queries: {tried_queries}
- If you propose any new_queries (exact Wikipedia page titles), you MUST set "action": "retry".
- If no new_queries are needed, set "action": "continue".
"""

    return f"""You are filtering research chunks BEFORE outline creation for topic: "{topic}"

RETRIEVED CHUNKS (summaries only, with IDs):
{chunks_formatted}

{links_ctx}

Task 1: Identify chunks to DELETE that are obviously irrelevant.

DELETE ONLY if:
1. Discusses a completely unrelated topic or domain
2. Navigation/metadata pages ("List of...", "Category:", "Index of...")
3. Background information with no connection to "{topic}"

KEEP if:
- Directly discusses "{topic}" or its main participants/entities
- Provides background context relevant to "{topic}"
- About key entities involved in "{topic}" (e.g., for "2022 Event X", keep pages about participants, venue, etc.)
- About the general topic domain

Task 2: NEW QUERIES MUST BE MORE SPECIFIC, NOT BROADER

If proposing new_queries:
- Queries MUST be MORE SPECIFIC subaspects of "{topic}"
- For temporal topics (with years/dates), queries MUST be about the SAME time period
- VALID: More specific aspects of the event itself
- INVALID: Broader histories, general categories, unrelated time periods

{retry_instruction}

Output format:
{{
    "action": "continue" or "retry",
    "delete_chunk_ids": ["id1", "id2", ...],
    "new_queries": ["Exact Wikipedia Page Title 1", "Exact Wikipedia Page Title 2"],
    "reasoning": "Brief explanation (1-2 sentences)"
}}
"""


def build_writer_tom_prediction_prompt(
    interaction_history: List[Dict[str, Any]],
    current_draft_info: dict,
) -> str:
    """
    Build Theory of Mind prediction prompt for Writer predicting Reviewer's next action.

    Args:
        interaction_history: List of all past observed agent actions
        current_draft_info: Dict with word_count, section_count, iteration

    Returns:
        Prompt for writer to predict reviewer's likely feedback focus
    """
    word_count = current_draft_info.get("word_count", "unknown")
    section_count = current_draft_info.get("section_count", "unknown")
    iteration = current_draft_info.get("iteration", 0)

    reviewer_actions = [
        "focus_on_accuracy",
        "focus_on_content_expansion",
        "focus_on_structure",
        "focus_on_clarity",
        "focus_on_style",
        "balanced_feedback",
    ]

    history_lines = []
    if interaction_history:
        for obs in interaction_history:
            history_lines.append(
                f"- Iteration {obs.get('iteration')}: {obs.get('agent')} observed doing '{obs.get('action')}'"
            )
    history_log = "\n".join(history_lines) if history_lines else "No history yet."

    return f"""You are a WRITER agent predicting the REVIEWER's next feedback focus.

CURRENT SITUATION:
- You are about to submit your draft for iteration {iteration}.
- Article word count: {word_count}
- Number of sections: {section_count}

INTERACTION HISTORY:
{history_log}

PREDICTION TASK:
Analyze the sequence of past interactions to identify patterns in the Reviewer's behavior.
Based on this history and the current article state, predict which aspect the Reviewer will focus their *next* round of feedback on.

AVAILABLE REVIEWER ACTIONS:
{reviewer_actions}

PREDICTION GUIDELINES:
- Consider the article's completeness (word count, sections).
- Consider any patterns in the Reviewer's past behavior.
- e.g., If the Reviewer focused on 'accuracy' last time, will they continue or move to 'style'?
- e.g., If the article is new (iteration 0), what is the most likely initial focus?

OUTPUT:
- predicted_action: ONE of the actions from the list above
- confidence: 0.0 to 1.0 based on how certain you are
- reasoning: 2-3 sentences explaining your prediction based on the history.

Your prediction will help you prepare the appropriate revisions."""


def build_revision_batch_prompt_v2(
    article: "Article",
    sections_to_revise: List[Dict[str, Any]],
    research_context: str,
    tom_context: Optional[str] = None,
) -> str:
    """
    Generate revision prompt for multiple sections with comprehensive context.

    Args:
        article: The article being revised
        sections_to_revise: List of dicts with keys:
            - section_name: str
            - section_content: str
            - section_summary: str
            - pending_feedback: List[FeedbackStoredModel]
            - resolved_feedback: List[FeedbackStoredModel]
        research_context: Formatted research chunks
        tom_context: Optional theory of mind context
    """
    sections_info = []
    for section_data in sections_to_revise:
        section_name = section_data["section_name"]
        section_content = section_data["section_content"]
        section_summary = section_data.get("section_summary", "")
        pending_feedback = section_data.get("pending_feedback", [])
        resolved_feedback = section_data.get("resolved_feedback", [])

        # Build feedback sections
        resolved_lines = []
        if resolved_feedback:
            resolved_lines.append(
                "RESOLVED FEEDBACK (for context - already addressed):"
            )
            for item in resolved_feedback:
                resolved_lines.append(f"  - {fmt_feedback_line(item)} [RESOLVED]")

        pending_lines = []
        if pending_feedback:
            pending_lines.append("PENDING FEEDBACK (must address in this revision):")
            for item in pending_feedback:
                pending_lines.append(f"  - {fmt_feedback_line(item)}")

        section_info = f"""=== Section: {section_name} ===
Summary: {section_summary}

Current text:
{section_content}

{chr(10).join(resolved_lines) if resolved_lines else ""}

{chr(10).join(pending_lines) if pending_lines else ""}"""

        sections_info.append(section_info)

    sections_text = "\n\n".join(sections_info)

    tom_section = ""
    if tom_context:
        tom_section = f"""
STRATEGIC CONTEXT (Theory of Mind):
{tom_context}

STRATEGIC GUIDANCE FOR WRITER:
Based on the predicted reviewer behavior, prioritize your revision approach:
- If reviewer likely to focus on accuracy: Prioritize fact-checking and citation accuracy; verify all claims against research
- If reviewer likely to focus on expansion: Add missing details and context; expand sections with available research
- If reviewer likely to focus on structure: Ensure logical flow and clear organization; improve transitions
- If reviewer likely to focus on clarity: Simplify complex sentences; ensure accessibility while maintaining accuracy
- If reviewer likely to focus on style: Polish tone and encyclopedic voice; ensure consistency

Use this prediction to preemptively address the most likely concerns in your revision.

"""

    return f"""
Revise {len(sections_to_revise)} sections of an article about "{article.title}" to address PENDING feedback.

{sections_text}

{tom_section}
RESEARCH CONTEXT:
{safe_trim(research_context)}

{CORE_TOPIC_FOCUS.format(topic=article.title)}
{CORE_NO_METACOMMENTARY}
{CORE_WRITING_RULES}
{CORE_CITATION_RULES}

CRITICAL REVISION PHILOSOPHY:
**EXPAND, DON'T CONDENSE**: When adding new information, PRESERVE existing factual details.
- If feedback requests adding missing information → ADD it while keeping current content
- If feedback identifies incorrect information → CORRECT it and ADD proper context
- ONLY remove content if feedback explicitly identifies it as wrong or irrelevant
- Aim to INCREASE section length when incorporating new research
- Think: "How can I make this MORE comprehensive?" not "How can I make this shorter?"

CRITICAL HONESTY REQUIREMENT:
For EACH feedback item you MUST:
1. READ the feedback carefully and understand what is being asked
2. MAKE THE CHANGE in the updated_content
3. SET STATUS HONESTLY:
   - Set 'addressed' ONLY if you genuinely made the requested change
   - Set 'wont_fix' if you cannot/should not make the change (explain why in writer_comment)
4. The reviewer WILL VERIFY your work - dishonest claims will be caught

DO NOT claim 'addressed' unless you actually did the work. Be honest.

BATCH OUTPUT RULES:
- Return a single JSON object matching WriterValidationBatchModel exactly
- **MANDATORY**: You MUST provide EXACTLY {len(sections_to_revise)} items in your response - one for EACH section listed above
- If a section is listed above, you MUST include an item for it in your response
- Only change the parts of the section that are necessary to address feedback, to make sure you preserve existing content as much as possible
- You may add new paragraphs, sentences, or details as needed to address feedback(It is encouraged to EXPAND content when adding new information)
- For each item:
  • updated_content MUST be the FULL replacement text for that section
  • START updated_content with the markdown H2 heading: ## Section Name
  • updates MUST include a status for EVERY feedback id listed for that section (addressed or wont_fix)
- Keep tone neutral, factual; no conversational preambles
- Preserve correct facts; NEVER invent citations or sources
- When adding information from research, INTEGRATE it naturally with existing content

Return EXACTLY this JSON shape with {len(sections_to_revise)} items:
{{
  "items": [
    {{
      "section_name": "<exact section name from above>",
      "updates": [{{"id":"<fid>","status":"addressed|wont_fix","writer_comment":"<note>"}}],
      "updated_content": "<full replacement text for this section>"
    }},
    ... (repeat for ALL {len(sections_to_revise)} sections)
  ]
}}
"""


# endregion Writer Prompt Templates
