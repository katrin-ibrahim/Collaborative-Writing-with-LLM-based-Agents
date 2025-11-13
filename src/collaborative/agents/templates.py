from typing import Dict, List, Optional

from src.collaborative.utils.models import FeedbackStoredModel
from src.utils.data import Article


# ---------------------------------------- Reviewer Templates ---------------------------------------
# region Reviewer Prompt Templates
def build_strategy_prompt(
    metrics: dict,
    validation_results: dict,
    iteration: int,
    prev_feedback_summary: dict,
) -> str:
    return f"""
You are a Reviewer planning the next review pass.

Context:
- Iteration: {iteration}
- Article metrics: {metrics}
- Citation validation: {validation_results}
- Previous feedback summary: {prev_feedback_summary}

Decide the BEST strategy for the next review pass.
Pick ONE:
- "citation-focused"  (if citations missing/invalid, sources needed, or verifiability is weak)
- "expansion-focused" (if article is short or lacks depth/coverage)
- "accuracy-focused"  (if there are correctness, clarity, contradiction, or OFF-TOPIC sections)
- "holistic"          (if the article is maturing and needs broad polish)

CRITICAL: If you see ANY sections that are off-topic or not directly relevant to the article title, use "accuracy-focused" strategy.
Sections must directly relate to the main topic - no tangential or weakly related content allowed.

Return JSON with:
{{
  "strategy": "citation-focused" | "expansion-focused" | "accuracy-focused" | "holistic",
  "rationale": "short reason",
  "focus_sections": ["optional", "section", "names"]
}}
""".strip()


def build_review_prompt_for_strategy(
    article: Article,
    metrics: Dict,
    validation_results: Dict,
    tom_context: Optional[str],
    strategy: str,
    research_context: Optional[str] = None,
    max_suggested_queries: int = 3,
) -> str:
    """Build strategy-aware review prompt with dynamic sections."""

    # Format metrics
    metrics_text = f"""
Article Metrics:
- Word count: {metrics.get('word_count', 'N/A')}
- Section count: {metrics.get('section_count', 'N/A')}
- Average section length: {metrics.get('avg_section_length', 'N/A')}
"""

    # Format citation validation results
    validation_text = f"""
Citation Validation:
- Total citations: {validation_results.get('total_citations', 0)}
- Valid citations: {validation_results.get('valid_citations', 0)}
- Missing chunks: {len(validation_results.get('missing_chunks', []))}
- Needs source tags: {validation_results.get('needs_source_count', 0)}
"""

    # Include research context if provided
    research_section = ""
    if research_context:
        research_section = f"""
CITED RESEARCH CHUNKS (for fact-checking):
The following chunks were CITED by the writer in the article. Use these to verify factual accuracy:

{research_context}

CRITICAL FACT-CHECKING INSTRUCTIONS:
1. Compare article claims against the chunk content provided above
2. Flag claims as "accuracy" issues if they are NOT supported by the cited chunks
3. Use "citation_missing" if factual claims lack citations entirely
4. Use "needs_source" if claims need additional supporting evidence
5. This is your PRIMARY tool for catching incorrect or unverified claims

Look for:
- Claims that contradict the chunk content
- Claims that go beyond what the chunks actually say
- Missing context or nuance from the source material
- Entities, facts, or details mentioned in chunks but missing from article
"""

    # Strategy-specific instructions
    strategy_instructions = {
        "citation-focused": """
FOCUS: Citation quality and completeness.

Your primary task is to assess citation quality using ONLY these feedback types:
- "citation_missing" - For claims that lack proper citations
- "citation_invalid" - For invalid or missing chunk IDs
- "needs_source" - For claims needing additional sourcing

Create individual feedback items (in the 'items' array) for each citation issue found.
DO NOT create items with types like 'citation_issues', 'citation_coverage', or 'citation_quality'.
""",
        "expansion-focused": """
FOCUS: Content gaps and structural improvements.

Your primary task is to identify areas for expansion using ONLY these feedback types:
- "content_expansion" - For sections that need more depth or missing topics
- "structure" - For organizational improvements
- "depth" - For areas lacking sufficient detail

Create individual feedback items (in the 'items' array) for each expansion opportunity.
DO NOT create items with types like 'coverage_gaps' or 'suggested_sections'.
""",
        "accuracy-focused": """
FOCUS: Factual accuracy and claim verification.

Your primary task is to ensure factual reliability by CHECKING CLAIMS AGAINST CITED CHUNKS:

Use ONLY these feedback types:
- "accuracy" - For claims that are UNSUPPORTED or CONTRADICTED by their cited chunks
- "needs_source" - For statements that need stronger evidence or additional citations
- "citation_missing" - For unsupported factual claims that lack any citation

CRITICAL: When you see cited chunks provided above, verify each major claim in the article:
1. Find the claim in the article
2. Check if it matches what the cited chunk actually says
3. Flag as "accuracy" if the chunk doesn't support the claim
4. Flag as "citation_missing" if there's no citation at all
5. Flag as "needs_source" if the citation is weak or insufficient

Create individual feedback items (in the 'items' array) for each accuracy concern.
DO NOT create items with types like 'accuracy_concerns' or 'factual_issues'.
BE CRITICAL - catch exaggerations, unsupported claims, and contradictions.
""",
    }

    # Build holistic strategy instruction with article title
    holistic_instruction = f"""
FOCUS: Overall article quality, coherence, and RELEVANCE to the topic.

CRITICAL RELEVANCE CHECK:
Before anything else, review EACH section title against the article title: "{article.title}".
- Does EVERY section directly relate to the main topic?
- Are there any tangential or weakly related sections?
- Would a reader expect to find this section in an article about this topic?

If you find OFF-TOPIC sections, create "relevance_focus" feedback items flagging them for removal or major revision.

Your task is to provide balanced, comprehensive feedback using ANY of these feedback types:
- "citation_missing", "citation_invalid", "needs_source" (for citations)
- "clarity", "accuracy", "structure" (for quality)
- "content_expansion", "redundancy", "tone", "depth", "flow" (for content)

Create individual feedback items (in the 'items' array) for each distinct issue.
Balance all aspects of article quality without focusing too heavily on any single type.

PRIORITY: Relevance and focus issues are CRITICAL - flag any off-topic content immediately.
"""

    # Add holistic to dictionary
    strategy_instructions["holistic"] = holistic_instruction

    # Build section list for feedback
    section_list = "\n".join([f"- {name}" for name in article.sections.keys()])

    # ToM context if available
    tom_section = ""
    if tom_context:
        tom_section = f"\n\nCollaborative Context (Theory of Mind):\n{tom_context}\n"

    # Suggested queries instruction
    suggested_queries_instruction = ""
    if research_context:
        suggested_queries_instruction = f"""

KNOWLEDGE GAP IDENTIFICATION:
If you identify factual gaps or missing entities that could be filled with additional research, populate the 'suggested_queries' field.
These should be specific, targeted Wikipedia-style search queries that would help the writer fill the gaps.

Rules for suggested_queries:
- Maximum {max_suggested_queries} queries per review
- Format as Wikipedia page titles (e.g., "Chris Scott (Australian footballer)", "UEFA Euro 2016 knockout phase")
- Focus on missing entities, events, or concepts that would strengthen the article
- Only suggest queries for HIGH-PRIORITY gaps that significantly impact article quality
- Leave empty if current research coverage is adequate

Example suggested_queries: ["Economic impact of 2020 Olympics", "Historical context of USMCA negotiations", "Biden climate policy timeline"]
"""

    return f"""You are reviewing an article using a {strategy} review strategy.

ARTICLE INFORMATION:
Title: {article.title}

Content:
{article.content}

{metrics_text}
{validation_text}
{research_section}

REVIEW STRATEGY:
{strategy_instructions[strategy]}
{tom_section}

ARTICLE SECTIONS:
{section_list}

REQUIRED OUTPUT FORMAT:
Your response must be a valid JSON object with the following structure:
{{
  "items": [
    {{
      "section": "section name or '_overall' for article-level feedback",
      "type": "MUST be exactly one of these 11 values: citation_missing, citation_invalid, needs_source, clarity, accuracy, structure, content_expansion, redundancy, tone, depth, flow",
      "issue": "clear description of what is wrong",
      "suggestion": "specific actionable recommendation to fix it",
      "priority": "high, medium, or low",
      "quote": "optional: exact text excerpt related to the issue",
      "paragraph_number": "optional: 1-indexed paragraph number",
      "location_hint": "optional: additional location context"
    }}
  ],
  "overall_assessment": "optional: holistic summary of the article quality",
  "suggested_queries": []
}}

CRITICAL TYPE CONSTRAINT:
The "type" field MUST be EXACTLY one of these 11 strings (no variations allowed):
1. citation_missing
2. citation_invalid
3. needs_source
4. clarity
5. accuracy
6. structure
7. content_expansion
8. redundancy
9. tone
10. depth
11. flow

DO NOT use: citation_issues, citation_coverage, coverage_gaps, accuracy_concerns, or any other variations.

GUIDELINES:
- Create one item per distinct issue found
- Assign priority based on impact: high (critical), medium (important), low (minor)
- Use the 'quote' field when referencing specific text
- Use '_overall' as section name for article-level feedback
- Be specific and actionable in your suggestions

Focus on providing constructive, actionable feedback that helps the writer improve the article.
{suggested_queries_instruction}
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


def outline_prompt(
    topic: str, top_chunk: str, chunk_summaries: str, temporal_context: str = ""
) -> str:
    """
    Generates the prompt for creating a structured article outline using LLM.

    Args:
        topic: The main subject of the article.
        top_chunk: The highest-ranked chunk from Phase 1 search for context.
        chunk_summaries: The formatted summaries of all stored research chunks.
        temporal_context: Optional context from adjacent year for disambiguation.

    Returns:
        The prompt string instructing the LLM to output a structured JSON outline.
    """

    # Consolidate research context
    research_context = consolidate_research(topic, top_chunk, chunk_summaries)

    # Add temporal context if provided (for disambiguation only, not for writing)
    temporal_disambiguation = ""
    if temporal_context:
        temporal_disambiguation = f"""
TEMPORAL DISAMBIGUATION CONTEXT (FOR UNDERSTANDING ONLY - DO NOT WRITE ABOUT THIS):
{temporal_context}

The above is from an adjacent year and is provided ONLY to help you understand the structure and category of "{topic}".
DO NOT create sections about the adjacent year. Use it only to understand what kind of event "{topic}" is.
"""

    # NOTE: The instruction for JSON output will be added by the call_structured_api method.
    # We focus here on the core task.

    return f"""
    Generate a Wikipedia-style article outline for the topic: "{topic}".

    CRITICAL TOPIC FOCUS:
    - This article MUST be specifically about "{topic}" - not related topics, background events, or similar subjects.
    - If the research context mentions related but different topics (e.g., different competitions, different years, different categories),
      DO NOT create sections about those topics. Focus ONLY on "{topic}" itself.
    - Every section heading must directly relate to "{topic}" - reject any tangential or background topics that distract from the main subject.
    - When in doubt: "Is this section specifically about '{topic}'?" If no, exclude it.

    {temporal_disambiguation}

    Use the research context to identify key entities such as people, organizations, events, locations, and outcomes.
    Headings should reflect these specific entities and capture factual, verifiable aspects of the topic.

    {research_context}

    Guidelines for Wikipedia-style section headings:
    - Each heading must refer to a concrete entity, event, or concept found in the context.
    - Prioritize entity-rich headings that include NAMES of people, places, organizations, or specific events.
      Examples: "Response by Pakistani Government", "Impact in Sindh Province", "International Aid from UN"
      (NOT: "Government Response", "Regional Impact", "International Aid")

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
    - DO NOT Return a DICT or ANYTHING other than a LIST of strings.
    """


def refine_outline_prompt(
    topic: str, top_chunk: str, chunk_summaries: str, previous_outline: str
) -> str:
    """Generate prompt for refining article outline with research context."""
    research_context = consolidate_research(topic, top_chunk, chunk_summaries)

    prompt = f"""
You are refining an existing article outline for the topic: "{topic}", to be more entity-rich and Wikipedia-style.
The outline should be informed by the research context provided below.

CRITICAL TOPIC FOCUS:
- This article MUST be specifically about "{topic}" - not related topics, background events, or similar subjects.
- Review the existing outline and REMOVE any sections that are NOT directly about "{topic}".
- Do not keep sections about related competitions, different years, background history, or tangential topics.
- Every section must pass the test: "Is this specifically about '{topic}'?" If no, remove it.

{research_context}

Existing Outline:
{previous_outline}

Your task is to enhance this outline by:
- Making section titles more specific and entity-rich (include names of people, places, organizations, specific events)
- Converting analytical/topical headings into factual, entity-focused ones
  Example transformations:
  • "Climate Factors" → "Monsoon Season in Pakistan"
  • "Government Response" → "Response by Pakistani Government"
  • "Environmental Impact" → "Impact in Sindh Province"
- Ensuring all sections leverage concrete entities from the research context
- Following Wikipedia conventions: "Background", "Impact in [Region]", "Response", "Aftermath" patterns
- Avoiding abstract/analytical titles like "Analysis", "Implications", "Causes and Effects"
- Maintaining exactly {HEADING_COUNT} sections
"""
    return prompt


def consolidate_research(topic: str, top_chunk: str, chunk_summaries: str) -> str:
    """Generate prompt for consolidating research information for article writing."""
    research_context = ""
    if top_chunk or chunk_summaries:
        research_context = "\n--- RESEARCH CONTEXT ---\n"
        if top_chunk:
            research_context += f"TOP CHUNK:\n{top_chunk}\n\n"
        if chunk_summaries:
            research_context += f"SUMMARIZED RESEARCH:\n{chunk_summaries}\n"
        research_context += "----------------------\n"

    return research_context


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
    import re

    year_match = re.search(r"\b(19|20)\d{2}\b", topic)
    topic_year = year_match.group(0) if year_match else None

    year_awareness = ""
    if topic_year:
        year_awareness = f"""
TEMPORAL AWARENESS:
The topic is about {topic_year}. When selecting chunks:
- STRONGLY PREFER chunks that mention {topic_year} or events from {topic_year}
- PENALIZE chunks about different years unless they provide essential background
- General background without year mentions is acceptable
"""

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

{year_awareness}

CRITICAL SELECTION RULES:

1. **PER SECTION ALLOCATION:** You must select UP TO {num_chunks} chunk IDs FOR EACH SECTION.
   - This means a TOTAL of up to {num_chunks * num_sections} chunks across all sections
   - Each section gets its own allocation of up to {num_chunks} chunks
   - DO NOT limit the total across all sections to {num_chunks}!

2. **Relevance:** For each section, select chunks most relevant to that specific section heading.

3. **Minimum:** Select AT LEAST ONE chunk per section (never return empty chunk_ids arrays).

4. **Maximum:** Select NO MORE than {num_chunks} chunks per section.

5. **Temporal Relevance:** If the topic has a year, prioritize chunks from that year.

6. **Output Format:** Return a JSON object with a "selections" array containing one object per section:
   {{
     "selections": [
       {{
         "section_heading": "exact heading from list",
         "chunk_ids": ["id1", "id2", "id3", ...]  // up to {num_chunks} IDs
       }},
       ...
     ]
   }}

7. **ID Integrity:** Use only the literal chunk IDs from the Research Context above.

8. **No Chatter:** Return ONLY the JSON object, no explanations.

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


def search_query_generation_prompt(
    topic: str, context: str = "", num_queries: int = 5
) -> str:
    """Generate prompt for creating targeted search queries for structured output."""

    # Extract year from topic if present (e.g., "2022 AFL Grand Final" -> 2022)
    import re

    year_match = re.search(r"\b(19|20)\d{2}\b", topic)
    topic_year = year_match.group(0) if year_match else None

    year_awareness_section = ""
    if topic_year:
        year_awareness_section = f"""
CRITICAL: The topic is about {topic_year}. Focus on:
- Events, people, and entities relevant to {topic_year}
- Do NOT query about different years unless getting background context
- Prefer specific {topic_year} entities over general historical articles
"""

    context_section = ""
    if context:
        context_section = f"""
CONTEXT FROM INITIAL SEARCH:
{context}

First, extract key entities from this context:
...
Then use these entities to create specific, relationship-focused queries.
"""

    return f""" You are generating search queries for a Wikipedia knowledge base.

TASK: Generate exactly {num_queries} Wikipedia article titles that would be useful for writing about "{topic}".

APPROACH:
1. Identify the key entities involved in "{topic}" based on your knowledge
2. For each entity, provide its Wikipedia article title
3. Format titles as they would appear on Wikipedia (proper capitalization, disambiguation in parentheses if needed)

{year_awareness_section}

{context_section}

GUIDELINES:
- Prefer specific named entities (people, teams, places, organizations, events) over generic topics
- Use disambiguation when needed: e.g., "Chris Scott (Australian footballer)"
- Keep titles concise (≤ 8 words)
- If topic contains a year, focus on entities from that specific year

You should return the titles under the key "queries" in a JSON object
"""


def extract_entities_from_context_prompt(topic: str, context: str) -> str:
    """Extract key entities from index/season page context to generate targeted queries."""
    return f"""You are extracting key entities from Wikipedia content to help write about "{topic}".

CONTEXT FROM BROADER WIKIPEDIA PAGE:
{context[:2000]}

TASK:
Based ONLY on the context above, identify the key entities (people, teams, organizations, locations, events) that are directly involved in "{topic}".

INSTRUCTIONS:
- Extract ONLY entities that are explicitly mentioned in the context
- Focus on entities that are central to "{topic}" itself
- Do NOT hallucinate or guess entities not in the context
- Return specific Wikipedia page titles for each entity
- Use proper capitalization and disambiguation (e.g., "Chris Scott (Australian footballer)")

Return a JSON object with key "entities" containing a list of Wikipedia page titles (max 5).

Example response format:
{{
  "entities": ["Entity Name 1", "Entity Name 2 (disambiguation)", "Entity Name 3"]
}}
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
    return f"{base_prompt}\n\nCollaborative context: {tom_context}"


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
    return f"""You are refining an encyclopedia article through internal self-correction.

ARTICLE TITLE: {title}

CURRENT DRAFT:
{current_text}

SELF-REFINEMENT TASK:
Perform internal quality improvement by:

1. CONTENT QUALITY:
   - Identify and fix factual inconsistencies or unclear statements
   - Ensure all claims are well-supported by the existing content
   - Remove redundant or repetitive information
   - Add transitions where flow is choppy

2. STRUCTURE & ORGANIZATION:
   - Verify logical section ordering
   - Ensure each paragraph has a clear purpose
   - Check that the article tells a coherent story

3. WRITING STYLE:
   - Replace vague or generic language with specific details
   - Eliminate clichés and overused phrases
   - Improve sentence variety and readability
   - Maintain encyclopedic tone (objective, informative)

4. TECHNICAL CORRECTNESS:
   - Fix grammar, spelling, and punctuation errors
   - Ensure consistent terminology throughout
   - Preserve all citation markers (e.g., <c cite="..."/>)
   - Maintain article length (within 10% of original)

CRITICAL CONSTRAINTS:
- Do NOT add new facts or information not in the original
- Do NOT remove factual content
- Do NOT change the fundamental structure (section headings)
- Do NOT include meta-commentary ("I have refined...", etc.)
- Output ONLY the refined article text

OUTPUT: The complete refined article with all improvements applied.
"""


def build_research_gate_prompt(
    topic: str,
    outline_str: str,
    batch_idx: int,
    total_batches: int,
    chunks_formatted: str,
    max_queries: int,
) -> str:
    """Generate prompt for research gate agent to gather more info."""

    # Extract year from topic for temporal filtering
    import re

    year_match = re.search(r"\b(19|20)\d{2}\b", topic)
    topic_year = year_match.group(0) if year_match else None

    year_filter_instruction = ""
    if topic_year:
        year_filter_instruction = f"""
                TEMPORAL FILTERING (CRITICAL):
                - The topic is about {topic_year}.
                - DELETE chunks that primarily discuss different years (e.g., {int(topic_year)-1}, {int(topic_year)+1}, {int(topic_year)+5}) unless they provide essential background.
                - KEEP chunks about {topic_year} or general timeless information.
                - When proposing new queries, focus on {topic_year}-specific entities and events.
                """

    return f"""
                You are the research gate for the Writer. Topic: {topic}

                CRITICAL TOPIC FOCUS:
                - This article is ONLY about "{topic}" - not related topics, similar events, or background subjects.
                - DELETE chunks that discuss different but related topics (e.g., different competitions, leagues, categories, or competitions with similar names).
                - KEEP only chunks that directly discuss "{topic}" itself.

                Article Outline:
                {outline_str}

                Reviewing batch {batch_idx + 1}/{total_batches} of retrieved chunks:
                {chunks_formatted}

                {year_filter_instruction}

                Task:
                - Identify chunks that do NOT support any section in the outline OR that are about different topics. List their ids in delete_chunk_ids.
                - ONLY propose new_queries if there are CRITICAL missing entities that were mentioned in the initial query generation.
                - DO NOT propose queries for new teams, people, or organizations that were not in the original research plan.

                Rules:
                - Be AGGRESSIVE with deletions — remove chunks about related but different topics (different leagues, years, categories, matches, etc.).
                - AGGRESSIVELY delete chunks about wrong years if the topic has a specific year.
                - DELETE chunks about similarly-named but different events or competitions.
                - DELETE chunks that mention other teams/competitors not relevant to this specific topic.
                - VERY RARELY propose new queries - only if there's an obvious gap for a known entity (venue, organization, key person).
                - DO NOT propose queries for teams/people found in retrieved chunks unless they were in the initial query plan.
                - Limit new_queries to {max_queries} precise queries targeting specific outline sections.
                - Each new_query must be formatted like a Wikipedia page title:
                    • Use Title Case (no underscores or all-caps).
                    • Prefer the core entity name (person, place, organization, season, etc.) rather than repeating the topic.
                    • Add parenthetical disambiguation if needed (e.g., "(2021 season)" or "(organization)").
                    • Keep titles concise and natural — around 2–6 words.
                - Keep reasoning short and actionable.
                - Set action="retry" if you recommend any changes (deletions or new queries), otherwise "continue".
                """


def build_outline_gate_prompt(topic: str, outline_str: str, ctx: Optional[str]) -> str:
    if ctx:
        prev_refinement_dec_ctx = f"You previously decided to refine the outline with the following reasoning:\n{ctx}\n"
    else:
        prev_refinement_dec_ctx = ""
    return f"""You just created an outline for the article:
                {topic} outline:
                {outline_str}
                {prev_refinement_dec_ctx}
                If a quick refinement (adding, merging, or reordering sections) would clearly improve the outline, set action="retry"; otherwise "continue".
                Give reasoning briefly, mentioning which sections or entities might need changes.
                Follow these rules:
                - Choose "retry" ONLY if:
                  • Important subtopics or entities mentioned in the research are missing.
                  • The logical order of sections is clearly confusing or incomplete.
                  • Key entities, concepts or time periods are missing.
                - Choose "continue" if the outline is coherent and covers all major aspects,
                  even if minor details could be improved.
                - Default to "continue" if uncertain.
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
