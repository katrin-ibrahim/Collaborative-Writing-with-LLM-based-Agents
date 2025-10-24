from typing import Dict, List, Optional

from src.utils.data import Article


# ---------------------------------------- Reviewer Templates ---------------------------------------
# region Reviewer Prompt Templates
def build_review_prompt_for_strategy(
    article: Article,
    metrics: Dict,
    validation_results: Dict,
    tom_context: Optional[str],
    strategy: str,
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

    # Strategy-specific instructions
    strategy_instructions = {
        "citation-focused": """
FOCUS: Citation quality and completeness.

Your primary task is to assess citation quality:
- Identify all claims that lack proper citations
- Flag invalid or missing chunk IDs
- Assess overall citation coverage
- Provide specific guidance on improving citations

**IMPORTANT**: Populate the 'citation_issues' field with a list of specific problems found.
Example: ["Section 'Background' paragraph 2 lacks citation for population claim", "Invalid chunk ID in 'Methods' section"]
""",
        "expansion-focused": """
FOCUS: Content gaps and structural improvements.

Your primary task is to identify areas for expansion:
- Suggest new sections that would improve coverage
- Identify existing sections that need more depth
- Highlight important topics not adequately covered
- Provide specific suggestions for content additions

**IMPORTANT**: Populate both 'suggested_sections' and 'coverage_gaps' fields with actionable lists.
Example suggested_sections: ["Economic Impact Analysis", "Historical Context"]
Example coverage_gaps: ["No discussion of environmental factors", "Missing stakeholder perspectives"]
""",
        "accuracy-focused": """
FOCUS: Factual accuracy and claim verification.

Your primary task is to ensure factual reliability:
- Flag claims that appear unverified or questionable
- Check for internal contradictions
- Assess the quality and appropriateness of sources
- Identify statements that need stronger evidence

**IMPORTANT**: Populate the 'accuracy_concerns' field with specific claims needing attention.
Example: ["Population figure in intro needs verification", "Date discrepancy between sections 2 and 4"]
""",
        "holistic": """
FOCUS: Overall article quality and coherence.

Your task is to provide balanced, comprehensive feedback:
- Assess structure, flow, and organization
- Evaluate content quality and depth
- Check citations and accuracy
- Consider readability and clarity
- Balance all aspects of article quality

The optional fields (citation_issues, suggested_sections, etc.) can be populated if relevant, but are not required for holistic reviews.
""",
    }

    # Build section list for feedback
    section_list = "\n".join([f"- {name}" for name in article.sections.keys()])

    # ToM context if available
    tom_section = ""
    if tom_context:
        tom_section = f"\n\nCollaborative Context (Theory of Mind):\n{tom_context}\n"

    return f"""You are reviewing an article using a {strategy} review strategy.

ARTICLE INFORMATION:
Title: {article.title}

Content:
{article.content}

{metrics_text}
{validation_text}

REVIEW STRATEGY:
{strategy_instructions[strategy]}
{tom_section}

SECTION-SPECIFIC FEEDBACK:
Provide detailed feedback for each of the following sections:
{section_list}

For each section, assign a priority level (high/medium/low) based on the severity of issues found.

REQUIRED OUTPUT FORMAT:
Your response must be a valid JSON object matching the ReviewFeedbackOutput schema:
- overall_assessment: string (your holistic assessment)
- section_feedback: array of objects with section_name, feedback, and priority
- review_strategy_used: "{strategy}"
- Optional fields based on strategy (citation_issues, suggested_sections, coverage_gaps, accuracy_concerns)

Focus on providing constructive, actionable feedback that helps the writer improve the article.
"""


# Add to templates.py (optional - the prompt is inline above, but can extract)


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
Analyze the current article and evaluate:

1. **Addressed Items**: Were the marked items actually fixed? Verify changes are present.
2. **Pending Items**: Why weren't these addressed? Are they difficult or were they overlooked?
3. **Won't Fix Items**: Are the reasons valid? Should these be reconsidered?
4. **Needs Clarification**: Can you provide the clarification requested?

Provide a concise summary (4-6 sentences) assessing:
- How well the writer responded overall
- Which addressed items were successfully implemented
- Which pending items are most important to tackle next
- Any concerns about won't-fix or clarification requests
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
    research_context = ""
    if top_chunk or chunk_summaries:
        research_context = "\n--- RESEARCH CONTEXT ---\n"
        if top_chunk:
            research_context += f"TOP CHUNK:\n{top_chunk}\n\n"
        if chunk_summaries:
            research_context += f"SUMMARIZED RESEARCH:\n{chunk_summaries}\n"
        research_context += "----------------------\n"

    # NOTE: The instruction for JSON output will be added by the call_structured_api method.
    # We focus here on the core task.

    return f"""
Generate a comprehensive, entity-rich article outline for the topic: "{topic}".
The outline MUST be fully informed by the research context provided below.

{research_context}

STRICT REQUIREMENTS FOR JSON OUTPUT:
1. 'title' field must match the topic exactly.
2. 'headings' field must be a list of exactly {HEADING_COUNT} section titles.
3. All {HEADING_COUNT} section titles must be highly specific, analytical, and entity-rich
   (leveraging names, dates, organizations, and concepts). ABSOLUTELY NO generic titles
   (e.g., "Introduction," "Conclusion," "Key Points," or "Summary").
"""


def select_section_chunks_prompt(
    section_heading: str, topic: str, chunk_summaries: str, num_chunks: int
) -> str:
    """Generate prompt for selecting relevant research chunks for a section."""

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

Strict Selection Rules:
1. **Relevance is Key:** Select only the chunk IDs that are *most* relevant and necessary to support the content of the section heading.
2. **Limit:** Select no more than {num_chunks} chunk IDs.
3. **Format:** Output MUST be a single JSON object that conforms strictly to the provided Pydantic schema (a list of chunk IDs).
4. **ID Integrity:** Use only the literal chunk IDs provided in the Research Context. Do not shorten, invent, or modify the IDs.
5. **No Chatter:** Do not include any text, reasoning, markdown, or preamble outside the required JSON object.
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


def search_query_generation_prompt(
    topic: str, context: str = "", num_queries: int = 5
) -> str:
    """Generate prompt for creating targeted search queries for structured output."""
    context_section = ""
    if context:
        # ... (rest of the context_section creation remains the same) ...
        context_section = f"""
CONTEXT FROM INITIAL SEARCH:
{context}

First, extract key entities from this context:
...
Then use these entities to create specific, relationship-focused queries.
"""

    return f"""
You must generate exactly {num_queries} highly targeted search queries for comprehensive research about: {topic}
{context_section}
Create queries that explore specific relationships and details by combining entities and concepts from the context.

Query Strategy:
1. ENTITY RELATIONSHIPS: How do key people/organizations interact?
2. TEMPORAL ANALYSIS: What happened before/during/after key events?
3. CAUSAL CONNECTIONS: What factors led to specific outcomes?
4. COMPARATIVE ANALYSIS: How does this compare to similar cases?
5. DETAILED MECHANICS: What are the specific processes/strategies involved?

Guidelines:
- Use specific names, dates, and locations from the context
- Focus on relationships between entities rather than isolated facts
- Ask for detailed analysis rather than basic information
- Target controversial or disputed aspects when relevant

---
IMPORTANT: Your entire response must be a single JSON object that strictly adheres to the provided schema. Do not include any explanation or extra text outside the JSON object.

The resulting list ('queries' key) MUST contain exactly {num_queries} elements.
"""


def enhance_prompt_with_tom(base_prompt: str, tom_context: Optional[str]) -> str:
    """Enhance prompt with Theory of Mind context if available."""
    if not tom_context:
        return base_prompt
    return f"{base_prompt}\n\nCollaborative context: {tom_context}"


# endregion Writer Prompt Templates
