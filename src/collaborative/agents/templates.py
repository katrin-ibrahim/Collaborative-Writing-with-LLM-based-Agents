from typing import Dict, List, Optional

from src.collaborative.utils.models import FeedbackStoredModel
from src.utils.data import Article

# ---------------------------------------- Reviewer Templates ---------------------------------------
# region Reviewer Prompt Templates


# ---------------------------------------- Reviewer Templates ---------------------------------------
# region Reviewer Prompt Templates


def build_review_prompt(
    article: Article,
    validation_results: Dict,
    tom_context: Optional[str],
    strategy: str,
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

    return f"""You are reviewing an article using a {strategy} review strategy.

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

# ---------------------------------------- Writer Templates ----------------------------------------
# region Writer Prompt Templates
HEADING_COUNT = 15  # Must match MAX_HEADING_COUNT in models.py


def outline_prompt(topic: str, chunk_summaries: str) -> str:
    """
    Generates the prompt for creating a structured article outline using LLM.

    Args:
        topic: The main subject of the article.
        chunk_summaries: The formatted summaries of all stored research chunks.

    Returns:
        The prompt string instructing the LLM to output a structured JSON outline.
    """
    return f"""
    Generate a Wikipedia-style article outline for the topic: "{topic}".

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

    RESEARCH CONTEXT:
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

    OUTPUT FORMAT:
    Return a JSON object with one field:
    - "headings": A list of exactly {HEADING_COUNT} section headings as strings

    Example: {{"headings": ["Background", "Teams and Players", "Match Summary", "Aftermath", ...]}}
    """


def select_section_chunks_prompt(
    section_heading: str, topic: str, chunk_summaries: str, num_chunks: int
) -> str:
    """Generate prompt for selecting relevant research chunks for a section."""

    # Extract year from topic if present for temporal awareness
    import re

    year_match = re.search(r"\b(19|20)\d{2}\b", topic)
    topic_year = year_match.group(0) if year_match else None

    year_awareness = ""
    if topic_year:
        year_awareness = f"""
TEMPORAL AWARENESS:
The topic is about {topic_year}. When selecting chunks:
- STRONGLY PREFER chunks that mention {topic_year} or events from {topic_year}
- PENALIZE chunks about different years (e.g., {int(topic_year)-1}, {int(topic_year)+1}) unless they provide essential background
- If chunks mention specific years, they should align with {topic_year}
- General background without year mentions is acceptable
"""

    # 1. Structure the research context clearly
    research_context = f"""
Research Context (Chunk Summaries):
---
{chunk_summaries}
---
"""

    # 2. Define the writing task and constraints, emphasizing JSON output
    task_description = f"""
You are acting as a Research Editor. Your task is to select the exact set of research chunks
required to write the section titled "{section_heading}" for the main article on "{topic}".

{year_awareness}

Strict Selection Rules:
1. **Relevance is Key:** Select only the chunk IDs that are *most* relevant and necessary to support the content of the section heading.
2. **Temporal Relevance:** If the topic has a specific year, prioritize chunks from that year over other years.
3. **Minimum Requirement:** You MUST select AT LEAST ONE chunk. Even if the match isn't perfect, choose the most relevant available chunks.
4. **Limit:** Select no more than {num_chunks} chunk IDs.
5. **Format:** Return a single JSON object with a key called chunk_ids, which is a list of string chunk IDs.
6. **ID Integrity:** Use only the literal chunk IDs provided in the Research Context. Do not shorten, invent, or modify the IDs.
7. **No Chatter:** Do not include any text, reasoning, markdown, or preamble outside the required JSON object.

CRITICAL: If you find no perfect matches, select the chunks that are most INDIRECTLY relevant or provide useful context.
Examples of indirect relevance:
- General background on the topic that could inform the section
- Related entities, events, or concepts mentioned in the section heading
- Chronological context if the section involves a specific time period

DO NOT return an empty list. Always select at least the best available option(s).
"""

    return f"{task_description}\n{research_context}"


def select_sections_chunks_batch_prompt(
    section_headings: List[str], topic: str, chunk_summaries: str, num_chunks: int
) -> str:
    """Generate prompt for batch selecting relevant research chunks for multiple sections."""

    sections_list = "\n".join(f"{i+1}. {h}" for i, h in enumerate(section_headings))
    num_sections = len(section_headings)

    research_context = f"""
Research Context (Chunk Summaries):
---
{chunk_summaries}
---
"""

    example_output = """
Example Output Format (for 3 sections with num_chunks=5):
{{
  "selections": [
    {{
      "section_heading": "Early life",
      "chunk_ids": ["chunk_1", "chunk_5", "chunk_12", "chunk_23", "chunk_45"]
    }},
    {{
      "section_heading": "Career",
      "chunk_ids": ["chunk_2", "chunk_8", "chunk_15", "chunk_28"]
    }},
    {{
      "section_heading": "Legacy",
      "chunk_ids": ["chunk_3", "chunk_9", "chunk_18", "chunk_31", "chunk_42"]
    }}
  ]
}}

NOTICE: Each section gets its OWN list of up to {num_chunks} chunks. This is PER SECTION, not total!
"""

    task_description = f"""
You are acting as a Research Editor selecting research chunks for an article on "{topic}".

TASK: Select research chunks for ALL {num_sections} sections listed below.

Sections ({num_sections} total):
{sections_list}

CRITICAL SELECTION RULES:

1. **Relevance:** For each section, select chunks most relevant to that specific section heading.

3. **Minimum:** Select AT LEAST ONE chunk per section (never return empty chunk_ids arrays).

4. **Maximum:** Select NO MORE than {num_chunks} chunks per section.

5. **Temporal Relevance:** If the topic has a year, prioritize chunks from that year, penalize chunks from different years.

6. **ID Integrity:** Use only the literal chunk IDs from the Research Context above.

7. **No Chatter:** Return ONLY the JSON object, no explanations.

{example_output}

If no perfect matches exist for a section, select the most indirectly relevant chunks.
"""

    return f"{task_description}\n{research_context}"


def write_section_content_prompt(
    section_heading: str, topic: str, relevant_info: str
) -> str:
    """Generate prompt for section content with research information."""
    research_context = ""
    if relevant_info:
        research_context = f"""Use this possibly relevant information gathered during research: {relevant_info}"""

    return f"""
Write a comprehensive section titled "{section_heading}" for an article about "{topic}".

CRITICAL TOPIC FOCUS:
- This section MUST be about "{topic}" specifically - NOT related topics, background events, or similar subjects.
- If the research information mentions related but different topics (e.g., different leagues, years, categories, or similarly-named events),
  DO NOT write about those topics. Extract ONLY information that directly relates to "{topic}".
- When in doubt: Does this fact apply to "{topic}" specifically? If no, exclude it.

{research_context}

CRITICAL: Write ONLY the article content. Do NOT include:
- Conversational language ("Okay, here's a section...")
- Questions to the user ("What is the overall article's focus?")
- Meta-commentary about the writing process
- Placeholder text or requests for clarification

WRITING STYLE - ENGAGING ENCYCLOPEDIA:
- Lead with concrete, specific details rather than generic statements
- Use precise numbers, dates, names, and facts from the research
- Avoid clichés like "highly anticipated", "storied", "remarkable", "significant"
- When describing events, focus on what actually happened rather than abstract qualities
- Build narrative through factual progression (what happened first, then what, then what)
- Let specific details create interest rather than relying on evaluative adjectives
- Include context that helps readers understand why events matter

Requirements:
- Write 2-3 well-structured paragraphs of encyclopedia-style content
- Ground your content in the provided research information
- Include specific details and examples from the sources
- Maintain an informative, objective tone
- Ensure smooth transitions and logical flow
- Avoid repetition of information from other sections
- Start directly with the content, no preamble or introduction

CITATION INSTRUCTIONS - CRITICAL:
- The research information above contains chunk IDs at the start of each piece of information (format: "chunk_id {{content: ...}}")
- When referencing information from the research chunks, cite using the EXACT chunk ID in citation tags: <c cite="chunk_id"/>
- For claims that need additional sourcing, mark with: <needs_source/>
- ABSOLUTELY FORBIDDEN: Making up chunk IDs, URLs, or references not in the research data
- ABSOLUTELY FORBIDDEN: Using generic citations like [1], [2], [source_1], [chunk_1]
- ABSOLUTELY FORBIDDEN: Inventing fake URLs or fake chunk IDs like "chunk_42", "chunk_77"
- Example: If research shows "search_supabase_faiss_1_9f3b58e9 {{content: The bridge was completed in 2018..., score: 0.95}}", cite as: "The bridge was completed in 2018 <c cite="search_supabase_faiss_1_9f3b58e9"/>."
- Multiple citations: "The incident occurred on July 8th <c cite="search_supabase_faiss_2_abc123"/> and caused damage <c cite="search_hybrid_0_def456"/>."
- If no research chunk IDs are provided, do NOT include any citations - write content without citation tags
- VERIFICATION: Before writing each citation, confirm the exact chunk ID exists in the research information above
"""


def write_sections_batch_prompt(sections_with_chunks: List[dict], topic: str) -> str:
    """
    Generate prompt for batch writing multiple sections at once.

    Args:
        sections_with_chunks: List of dicts with 'section_heading' and 'relevant_info' keys
        topic: The article topic
    """
    sections_info = []
    for i, item in enumerate(sections_with_chunks, 1):
        section_heading = item["section_heading"]
        relevant_info = item.get("relevant_info", "")
        sections_info.append(
            f"""
=== Section {i}: {section_heading} ===
Research chunks for this section:
{relevant_info if relevant_info else "No specific research chunks assigned."}
"""
        )

    sections_text = "\n".join(sections_info)

    return f"""
Write encyclopedia-style content for {len(sections_with_chunks)} sections of an article about "{topic}".

{sections_text}

WRITING REQUIREMENTS:
1. For EACH section above, write 2-3 well-structured paragraphs
2. Use ONLY the research chunks provided for that specific section
3. Write in an informative, objective, encyclopedia tone
4. Start directly with content - NO conversational text, questions, or meta-commentary
5. Avoid repetition between sections

WRITING STYLE - ENGAGING ENCYCLOPEDIA:
- Lead with concrete, specific details rather than generic statements
- Use precise numbers, dates, names, and facts from the research
- Avoid clichés like "highly anticipated", "storied", "remarkable", "significant"
- When describing events, focus on what actually happened rather than abstract qualities
- Build narrative through factual progression (what happened first, then what, then what)
- Let specific details create interest rather than relying on evaluative adjectives
- Include context that helps readers understand why events matter

CITATION RULES - CRITICAL:
• Research chunks start with their ID (e.g., "abc123 {{content: ...}}")
• Cite using EXACT chunk ID in tags: <c cite="abc123"/>
• For claims needing sources: <needs_source/>
• NEVER invent chunk IDs not in the research above
• NEVER use placeholders like [1], chunk_id_1, source_1, etc.
• If no research provided for a section, write WITHOUT citations
• Example: "The event occurred in 2022 <c cite="947265bb2a2267de817bdef27b97a6dd"/>."

OUTPUT FORMAT (JSON):
{{
  "sections": [
    {{
      "section_heading": "exact heading from above",
      "content": "2-3 paragraphs of content here"
    }}
  ]
}}

Write content for ALL {len(sections_with_chunks)} sections listed above.
"""


def write_full_article_prompt(
    topic: str, headings: List[str], relevant_info: str
) -> str:
    """Generate prompt for full-article writing consistent with section prompt rules."""
    headings_hint = (
        "\n".join(f"{i+1}. {h}" for i, h in enumerate(headings)) if headings else ""
    )
    research_context = ""
    if relevant_info:
        research_context = f"Use this possibly relevant information gathered during research: {relevant_info}"

    example_format = f"""
EXAMPLE OUTPUT FORMAT:
## {headings[0] if headings else "First Section"}

Paragraph 1 content here with proper encyclopedia-style writing...

Paragraph 2 continues with more details and examples...

Paragraph 3 wraps up this section with additional context...

## {headings[1] if len(headings) > 1 else "Second Section"}

Paragraph 1 of second section begins here...

CRITICAL: Notice that EACH section must:
1. Start with EXACTLY "## " followed by the heading text
2. Have 2-4 paragraphs of content after the heading
3. Use the EXACT heading text from the outline below
"""

    return f"""
Write a high-quality encyclopedia-style article about "{topic}".

You MUST structure the article with the following sections in this exact order:
{headings_hint}

{example_format}

MANDATORY FORMATTING RULES (FAILURE TO FOLLOW WILL RESULT IN UNUSABLE OUTPUT):

1. **Section Headers:** EVERY section MUST start with its exact heading prefixed by '## ' (Markdown H2).
   - Correct: "## Early life and education"
   - WRONG: "Early life and education" (missing ##)
   - WRONG: "# Early life" (wrong number of #)
   - WRONG: "## Early Life" (different capitalization)

2. **Section Content:** After each ## heading, write 2–4 coherent paragraphs.

3. **Complete Coverage:** Include ALL {len(headings)} sections from the outline above.

4. **No Extras:** Do NOT add extra sections or change heading wording.

5. **Start Immediately:** Begin directly with the first ## heading. NO title, NO preamble.

{research_context}

CRITICAL: Write ONLY the article content. Do NOT include:
- Conversational language or meta-commentary
- Questions to the user
- Placeholder text or requests for clarification
- A separate title line (the first ## is the first section)

Requirements:
- Ground content in the provided research information where applicable
- Include specific details and examples from sources
- Maintain an informative, objective tone with smooth transitions
- Avoid repetition and avoid any preambles or summaries outside the article text

CITATION INSTRUCTIONS - CRITICAL:
- The research information above contains chunk IDs at the start of each piece of information (format: "chunk_id {{content: ...}}")
- When referencing information from the research chunks, cite using the EXACT chunk ID in citation tags: <c cite="chunk_id"/>
- For claims that need additional sourcing, mark with: <needs_source/>
- ABSOLUTELY FORBIDDEN: Making up chunk IDs, URLs, or references not in the research data
- ABSOLUTELY FORBIDDEN: Using generic citations like [1], [2], [source_1], [chunk_1]
- ABSOLUTELY FORBIDDEN: Inventing fake URLs or fake chunk IDs like "chunk_42", "chunk_77"
- If no research chunk IDs are provided, do NOT include any citations - write content without citation tags

OUTPUT CHECKLIST:
✓ Every section starts with ## followed by exact heading
✓ {len(headings)} sections total
✓ 2-4 paragraphs per section
✓ No title line before first section
✓ Citations use <c cite="chunk_id"/> format only
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


def enhance_prompt_with_tom(base_prompt: str, tom_context: Optional[str]) -> str:
    """Enhance prompt with Theory of Mind context if available."""
    if not tom_context:
        return base_prompt
    return f"{base_prompt}\n\n{tom_context}"


def build_directive_tom_context_for_writer(
    predicted_action: str, confidence: float, reasoning: str
) -> str:
    """Build directive ToM context for writer based on predicted reviewer behavior."""
    action_to_strategy = {
        "provide_extensive_feedback": (
            "The reviewer will provide extensive, detailed feedback.",
            "Focus on thoroughness and detail in your revisions. Expect comprehensive suggestions across multiple dimensions.",
        ),
        "provide_moderate_feedback": (
            "The reviewer will provide moderate, balanced feedback.",
            "Maintain a balance between depth and breadth in your revisions. Address key issues without over-engineering.",
        ),
        "approve_with_minor_suggestions": (
            "The reviewer will mostly approve with only minor suggestions.",
            "Focus on polishing and refinement. Small adjustments will likely be sufficient.",
        ),
        "focus_on_accuracy_refinement": (
            "The reviewer will focus primarily on factual accuracy and citations.",
            "Prioritize fact-checking, adding citations, and ensuring claims are well-supported. Accuracy is paramount.",
        ),
        "request_major_expansion": (
            "The reviewer will request significant content expansion.",
            "Be prepared to substantially expand content. Consider adding new sections or significantly deepening existing ones.",
        ),
    }

    prediction, strategy = action_to_strategy.get(
        predicted_action,
        (
            f"The reviewer will likely take action: {predicted_action}",
            "Adapt your approach based on reviewer expectations.",
        ),
    )

    confidence_level = (
        "high" if confidence >= 0.75 else "moderate" if confidence >= 0.5 else "low"
    )

    return f"""COLLABORATIVE INTELLIGENCE (Theory of Mind):
Based on past interactions, I predict: {prediction}
Confidence: {confidence_level} ({confidence:.1%})
Reasoning: {reasoning}

STRATEGIC ADAPTATION:
{strategy}

Use this prediction to proactively shape your work to align with reviewer expectations."""


def build_directive_tom_context_for_reviewer(
    predicted_action: str, confidence: float, reasoning: str
) -> str:
    """Build directive ToM context for reviewer based on predicted writer behavior."""
    action_to_strategy = {
        "accept_most_feedback": (
            "The writer will accept most of your feedback.",
            "Be comprehensive and detailed in your feedback. The writer is receptive and will implement most suggestions.",
        ),
        "partially_accept_maintain_creative_vision": (
            "The writer will partially accept feedback while maintaining their creative vision.",
            "Focus on high-priority issues and provide clear justification. The writer may selectively implement suggestions.",
        ),
        "contest_some_feedback": (
            "The writer may contest or reject some feedback.",
            "Ensure your feedback is well-justified and specific. Be prepared for pushback on lower-priority items.",
        ),
        "request_justification": (
            "The writer will request additional justification for feedback.",
            "Provide detailed rationale and specific examples. Anticipate questions about your suggestions.",
        ),
    }

    prediction, strategy = action_to_strategy.get(
        predicted_action,
        (
            f"The writer will likely take action: {predicted_action}",
            "Adjust your feedback approach based on writer tendencies.",
        ),
    )

    confidence_level = (
        "high" if confidence >= 0.75 else "moderate" if confidence >= 0.5 else "low"
    )

    return f"""COLLABORATIVE INTELLIGENCE (Theory of Mind):
Based on past interactions, I predict: {prediction}
Confidence: {confidence_level} ({confidence:.1%})
Reasoning: {reasoning}

STRATEGIC ADAPTATION:
{strategy}

Calibrate your feedback style and volume to maximize collaborative effectiveness."""


def build_single_section_revision_prompt(
    article: "Article",
    section: str,
    items: list["FeedbackStoredModel"],
    research_ctx: str,
) -> str:
    current_text = (
        article.content
        if section == "_overall"
        else (article.sections or {}).get(section, "")
    )
    lines: list[str] = [
        "ROLE: You are revising an encyclopedia-style article section.",
        "GOAL: Address ONLY the PENDING feedback for THIS section.",
        "",
        "CRITICAL HONESTY REQUIREMENT:",
        "For EACH feedback item you MUST:",
        "1. QUOTE the feedback: Read it carefully and understand what is being asked",
        "2. MAKE THE CHANGE: Actually implement the requested change in the updated_content",
        "3. SET STATUS HONESTLY:",
        "   - Set 'addressed' ONLY if you genuinely made the requested change",
        "   - Set 'wont_fix' if you cannot or should not make the change (explain why in writer_comment)",
        "4. The reviewer WILL VERIFY your work - dishonest claims will be caught",
        "",
        "DO NOT claim 'addressed' unless you actually did the work. Be honest.",
        "",
        "OUTPUT RULES:",
        "- Return a single JSON object matching WriterValidationModel exactly.",
        "- content_type MUST be 'partial_section'.",
        "- If section is '_overall', updated_content MUST be the FULL article with title heading (# Title) and all section headings (## Section Name).",
        "- If section is NOT '_overall', updated_content MUST be ONLY that section's text, starting with the markdown H2 heading (## Section Name).",
        "- updates MUST include a status for EVERY feedback id listed (addressed or wont_fix).",
        "- Keep tone neutral, factual, precise; no conversational preambles.",
        "- Preserve correct facts; NEVER invent citations or sources.",
        "- If a suggestion is unsafe or clearly wrong, set status='wont_fix' and explain shortly in writer_comment.",
        "- Keep length roughly similar unless feedback asks otherwise.",
        "",
        f"ARTICLE_TITLE: {article.title}",
        f"SECTION: {section}",
        f"RESEARCH_CONTEXT: {safe_trim(research_ctx)}",
        "CURRENT_TEXT:",
        safe_trim(current_text),
        "PENDING_FEEDBACK_ITEMS:",
    ]
    for it in items:
        lines.append(f"- {fmt_feedback_line(it)}")

    # strict schema echo
    lines += [
        "",
        "Return EXACTLY this JSON shape (no extra keys):",
        "{",
        '  "updates": [',
        '    {"id":"<fid>","status":"addressed|wont_fix","writer_comment":"<short note or empty>"}',
        "  ],",
        '  "updated_content": "<full replacement text for this section>",',
        '  "content_type": "partial_section"',
        "}",
    ]
    return "\n".join(lines)


def build_revision_batch_prompt(
    article: "Article",
    pending_by_section: dict[str, list["FeedbackStoredModel"]],
    research_ctx: str,
) -> str:
    lines: list[str] = [
        "ROLE: You are revising an encyclopedia-style article.",
        "GOAL: Address ONLY the PENDING feedback for the sections listed below.",
        "",
        "CRITICAL HONESTY REQUIREMENT:",
        "For EACH feedback item you MUST:",
        "1. QUOTE the feedback: Read it carefully and understand what is being asked",
        "2. MAKE THE CHANGE: Actually implement the requested change in the updated_content",
        "3. SET STATUS HONESTLY:",
        "   - Set 'addressed' ONLY if you genuinely made the requested change",
        "   - Set 'wont_fix' if you cannot or should not make the change (explain why in writer_comment)",
        "4. The reviewer WILL VERIFY your work - dishonest claims will be caught",
        "",
        "DO NOT claim 'addressed' unless you actually did the work. Be honest.",
        "",
        "OUTPUT RULES:",
        "- Return a single JSON object matching WriterValidationBatchModel exactly.",
        "- Provide ONE item per section that appears below.",
        "- For each item:",
        "  - content_type MUST be 'partial_section'.",
        "  - updated_content MUST be the FULL replacement text for that section.",
        "  - If the section is NOT '_overall', START updated_content with the markdown H2 heading: ## Section Name",
        "  - updates MUST include a status for EVERY feedback id listed for that section (addressed or wont_fix).",
        "- Keep tone neutral, factual; no conversational preambles.",
        "- Preserve correct facts; NEVER invent citations or sources.",
        "- If a suggestion is unsafe or clearly wrong, set status='wont_fix' and explain briefly.",
        "- Keep length roughly similar unless feedback asks otherwise.",
        "",
        f"ARTICLE_TITLE: {article.title}",
        f"RESEARCH_CONTEXT: {safe_trim(research_ctx)}",
        "",
        "SECTIONS_TO_REVISE:",
    ]

    for sec, items in pending_by_section.items():
        cur_text = (
            article.content
            if sec == "_overall"
            else (article.sections or {}).get(sec, "")
        )
        lines.append(f"\n=== SECTION: {sec} ===")
        lines.append("CURRENT_TEXT:")
        lines.append(safe_trim(cur_text, max_chars=6000))
        lines.append("PENDING_FEEDBACK_ITEMS:")
        for it in items:
            lines.append(f"- {fmt_feedback_line(it)}")

    # strict schema echo
    lines += [
        "",
        "Return EXACTLY this JSON shape (no extra keys):",
        "{",
        '  "items": [',
        "    {",
        '      "updates": [',
        '        {"id":"<fid>","status":"addressed|wont_fix","writer_comment":"<short note or empty>"}',
        "      ],",
        '      "updated_content": "<full replacement for this section>",',
        '      "content_type": "partial_section"',
        "    }",
        "  ]",
        "}",
    ]
    return "\n".join(lines)


def build_self_refine_prompt(title: str, current_text: str) -> str:
    """
    Generate prompt for self-refinement (internal polishing).
    This represents the self-correction approach where the writer
    internally improves their work without external critique.
    """
    return f"""You are refining an encyclopedia article through careful internal polishing.

ARTICLE TITLE: {title}

CURRENT DRAFT:
{current_text}

SELF-REFINEMENT TASK:
Perform LIGHT polishing to improve quality WITHOUT removing content:

1. FACTUAL CONSISTENCY (PRIMARY FOCUS):
   - Check if the article contains any contradicting statements
   - If two statements directly conflict, keep the more specific/supported one
   - Example: If article says both "Event occurred in 2022" and "Event occurred in 2023", resolve the conflict
   - DO NOT remove content just because it seems uncertain - only remove clear contradictions

2. LANGUAGE & FLOW:
   - Improve transitions between paragraphs
   - Fix awkward phrasing or unclear sentences
   - Enhance readability while preserving all information
   - Maintain encyclopedic tone (objective, informative)

3. TECHNICAL POLISH:
   - Fix grammar, spelling, and punctuation errors
   - Ensure consistent terminology throughout
   - Preserve all citation markers (e.g., <c cite="..."/>)
   - Remove any non-English text unless it's proper nouns

CRITICAL CONSTRAINTS:
- PRESERVE ALL SECTIONS: Do not delete or merge sections
- PRESERVE ALL FACTS: Do not remove information unless it directly contradicts other facts
- ADD NOTHING NEW: Do not add facts or information not in the original
- LIGHT TOUCH: Focus on polishing, not rewriting
- When in doubt, KEEP the content rather than removing it

This is a POLISH step, not an editorial review. The goal is to improve presentation of existing content, not to restructure or reduce it.
- DO resolve factual conflicts by choosing the most relevant/supported claim
- Do NOT change section headings
- Do NOT include meta-commentary ("I have refined...", etc.)
- Output ONLY the refined article text
- Article length may decrease if removing contradictions/redundancy

PRIORITY: Factual consistency and relevance are more important than length.

OUTPUT: The complete refined article with all improvements applied.
"""


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


# endregion Writer Prompt Templates


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
