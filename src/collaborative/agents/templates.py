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
- "accuracy-focused"  (if there are correctness, clarity, or contradiction issues)
- "holistic"          (if the article is maturing and needs broad polish)

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
AVAILABLE RESEARCH CONTEXT:
The following research chunks were used by the writer. Use this to identify factual gaps and missing information:

{research_context}

NOTE: You can now perform true fact-checking by comparing article claims against this research context.
Identify specific entities, facts, or claims that are missing or need additional sourcing.
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

Your primary task is to ensure factual reliability using ONLY these feedback types:
- "accuracy" - For claims that appear unverified or questionable
- "needs_source" - For statements needing stronger evidence
- "citation_missing" - For unsupported factual claims

Create individual feedback items (in the 'items' array) for each accuracy concern.
DO NOT create items with types like 'accuracy_concerns' or 'factual_issues'.
""",
        "holistic": """
FOCUS: Overall article quality and coherence.

Your task is to provide balanced, comprehensive feedback using ANY of these feedback types:
- "citation_missing", "citation_invalid", "needs_source" (for citations)
- "clarity", "accuracy", "structure" (for quality)
- "content_expansion", "redundancy", "tone", "depth", "flow" (for content)

Create individual feedback items (in the 'items' array) for each distinct issue.
Balance all aspects of article quality without focusing too heavily on any single type.
""",
    }

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
"""


def format_metrics(metrics: Dict) -> str:
    """Format metrics for display."""
    return f"""
- Word count: {metrics.get('word_count', 0)}
- Section count: {metrics.get('section_count', 0)}
- Avg section length: {metrics.get('avg_section_length', 0)}
"""


def format_validation(validation_results: Dict) -> str:
    """Format validation results for display."""
    missing = len(validation_results.get("missing_chunks", []))
    return f"""
- Total citations: {validation_results.get('total_citations', 0)}
- Valid: {validation_results.get('valid_citations', 0)}
- Missing chunks: {missing}
- Needs source: {validation_results.get('needs_source_count', 0)}
"""


# endregion Reviewer Prompt Templates

# ---------------------------------------- Writer Templates ----------------------------------------
# region Writer Prompt Templates
HEADING_COUNT = 6  # Must match MAX_HEADING_COUNT in models.py


def outline_prompt(topic: str, top_chunk: str, chunk_summaries: str) -> str:
    """
    Generates the prompt for creating a structured article outline using LLM.

    Args:
        topic: The main subject of the article.
        top_chunk: The highest-ranked chunk from Phase 1 search for context.
        chunk_summaries: The formatted summaries of all stored research chunks.

    Returns:
        The prompt string instructing the LLM to output a structured JSON outline.
    """

    # Consolidate research context
    research_context = consolidate_research(topic, top_chunk, chunk_summaries)

    # NOTE: The instruction for JSON output will be added by the call_structured_api method.
    # We focus here on the core task.

    return f"""
    Generate a Wikipedia-style article outline for the topic: "{topic}".

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


def write_section_content_prompt(
    section_heading: str, topic: str, relevant_info: str
) -> str:
    """Generate prompt for section content with research information."""
    research_context = ""
    if relevant_info:
        research_context = f"""Use this possibly relevant information gathered during research: {relevant_info}"""

    return f"""
Write a comprehensive section titled "{section_heading}" for an article about "{topic}".

{research_context}

CRITICAL: Write ONLY the article content. Do NOT include:
- Conversational language ("Okay, here's a section...")
- Questions to the user ("What is the overall article's focus?")
- Meta-commentary about the writing process
- Placeholder text or requests for clarification

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


def write_full_article_prompt(
    topic: str, headings: List[str], relevant_info: str
) -> str:
    """Generate prompt for full-article writing consistent with section prompt rules."""
    headings_hint = "\n".join(f"- {h}" for h in headings) if headings else ""
    research_context = ""
    if relevant_info:
        research_context = f"Use this possibly relevant information gathered during research: {relevant_info}"

    return f"""
Write a high-quality encyclopedia-style article about "{topic}".

You MUST structure the article to exactly fill the provided outline and include each section heading as a Markdown H2 line in this exact order:
{headings_hint}

Formatting rules:
- Begin each section with its exact heading prefixed by '## ' (Markdown H2).
- After each H2, write 2–4 coherent paragraphs for that section.
- Do not add extra sections or change heading wording.

{research_context}

CRITICAL: Write ONLY the article content. Do NOT include:
- Conversational language or meta-commentary
- Questions to the user
- Placeholder text or requests for clarification

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
- VERIFICATION: Before writing each citation, confirm the exact chunk ID exists in the research information above
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

    return f""" You are generating wiki-style search queries for knowledge retrieval.

TASK
Generate exactly {num_queries} concise Wikipedia page titles related to the topic below.
Titles must look like real Wikipedia article names — not questions.

INPUT TOPIC: "{topic}"

{year_awareness_section}

{context_section}

INSTRUCTIONS
- Prefer concrete, named entities (people, teams, venues, events, years).
- Use Wikipedia disambiguation style when needed: e.g., "Chris Scott (Australian footballer)".
- Avoid generic phrases like: result(s), history, finals series, overview, introduction, guide.
- ≤ 8 words per title. No punctuation except parentheses or hyphens.
- If topic contains a year, prioritize entities from that specific year.

EXAMPLES
Topic: "UEFA Euro 2016 Final"
Good → ["Portugal national football team", "France national football team", "Stade de France", "Éder (footballer, born 1987)", "UEFA Euro 2016 knockout phase"]
Bad  → ["Euro 2016 Final Result", "Euro 2016 History"]

Topic: "2022 AFL Grand Final"
Good → ["Geelong Football Club", "Sydney Swans", "Melbourne Cricket Ground", "2022 AFL season", "Joel Selwood"]
Bad  → ["AFL Grand Final", "2023 AFL Grand Final", "Grand Final history"]

Topic: "NASA Mars Perseverance"
Good → ["Perseverance (rover)", "Mars 2020", "Ingenuity (helicopter)", "Jezero crater", "Mars Sample Return"]

You should return the titles under the key "queries" in a JSON object


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
    Ultra-light full-article polish: clarity/grammar/flow only.
    """
    return (
        "ROLE: Lightly polish an encyclopedia article for clarity, grammar, and flow.\n"
        "RULES:\n"
        "- Do NOT change meaning, facts, or add new information.\n"
        "- Preserve any citation markers like [id].\n"
        "- Keep structure and length roughly the same.\n"
        "- Output ONLY the refined article text (no headers, no commentary).\n\n"
        f"ARTICLE_TITLE: {title}\n"
        "CURRENT_TEXT:\n"
        f"{current_text}\n"
    )


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

                Article Outline:
                {outline_str}

                Reviewing batch {batch_idx + 1}/{total_batches} of retrieved chunks:
                {chunks_formatted}

                {year_filter_instruction}

                Task:
                - Identify chunks that do NOT support any section in the outline. List their ids in delete_chunk_ids.
                - Identify outline sections lacking sufficient supporting chunks. Propose targeted new_queries to fill gaps.

                Rules:
                - Be conservative with deletions — only remove chunks clearly irrelevant to ALL outline sections.
                - AGGRESSIVELY delete chunks about wrong years if the topic has a specific year.
                - Limit new_queries to {max_queries} precise queries targeting specific outline sections.
                - Each new_query must be formatted like a Wikipedia page title:
                    • Use Title Case (no underscores or all-caps).
                    • Prefer the core entity name (person, place, organization, season, etc.) rather than repeating the topic.
                    • Add parenthetical disambiguation if needed (e.g., "(2021 season)" or "(organization)").
                    • Keep titles concise and natural — around 2–6 words.
                - Keep reasoning short and actionable.
                - Set action="retry" if you recommend any changes, otherwise "continue".
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
