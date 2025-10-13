# ---------------------------------------- Reviwer Templates ---------------------------------------
# src/collaborative/agents/templates/reviewer_prompts.py
"""
Prompt template functions for ReviewerAgent workflow.
"""

from typing import Dict, Optional

from src.utils.data import Article


def build_holistic_review_prompt(
    article: Article,
    metrics: Dict,
    fact_check_results: Dict,
    tom_context: Optional[str],
    review_strategy: str,
) -> str:
    """Build prompt for holistic review with section-specific feedback."""

    # Build metrics summary
    metrics_text = f"""
Article Metrics:
- Word count: {metrics.get('word_count', 'N/A')}
- Section count: {metrics.get('section_count', 'N/A')}
- Average section length: {metrics.get('avg_section_length', 'N/A')}
"""

    # Build fact-check summary
    verified_claims = fact_check_results.get("verified_claims", [])
    unverified_claims = fact_check_results.get("unverified_claims", [])

    fact_check_text = f"""
Fact-checking Results:
- Verified claims: {len(verified_claims)}
- Unverified claims: {len(unverified_claims)}
"""

    if unverified_claims:
        fact_check_text += f"Unverified claims that need attention:\n"
        for claim in unverified_claims[:3]:  # Limit to avoid overwhelming
            fact_check_text += f"- {claim}\n"

    # Build review strategy guidance
    strategy_guidance = _get_strategy_guidance(review_strategy)

    # Build collaborative context
    tom_guidance = ""
    if tom_context:
        tom_guidance = f"\n\nCollaborative Context: {tom_context}\n"

    return f"""Review this article using a {review_strategy} approach. Provide holistic analysis and section-specific feedback.

Title: {article.title}

Content:
{article.content}

{metrics_text}
{fact_check_text}

Review Strategy: {strategy_guidance}
{tom_guidance}

Provide your review in this format:

OVERALL ASSESSMENT:
[Your holistic assessment of the article]

SECTION FEEDBACK:
{build_section_feedback_template(article)}

Focus on providing constructive, specific feedback that helps improve the article according to the {review_strategy} strategy."""


def _get_strategy_guidance(strategy: str) -> str:
    """Get guidance text for review strategy."""
    guidance_map = {
        "expansion-focused": "Focus on areas where the article needs more depth, detail, or additional sections. Identify gaps in coverage and suggest specific expansions.",
        "accuracy-focused": "Prioritize fact-checking, source verification, and accuracy of claims. Focus on correcting any inaccuracies and improving factual reliability.",
        "holistic": "Provide comprehensive feedback covering structure, content quality, flow, and overall coherence. Balance all aspects of article quality.",
        "convergence-focused": "Focus on addressing previous unresolved feedback and ensuring the article is moving toward completion. Prioritize outstanding issues.",
    }
    return guidance_map.get(strategy, "Provide comprehensive, balanced feedback.")


def build_section_feedback_template(article: Article) -> str:
    """Build template for section feedback response using outline-style formatting."""
    template = "For each section below, provide specific feedback using the EXACT format shown:\n\n"
    for section_name in article.sections.keys():
        template += f"## {section_name}\n[Provide detailed feedback for the {section_name} section here]\n\n"

    template += "\nIMPORTANT: Use the exact markdown format ## Section Name (with double hashtags, space, then section name) for each section header."
    return template


# ---------------------------------------- Writer Templates ----------------------------------------
"""
Prompt template functions for WriterAgent workflow.
"""


def outline_prompt(topic: str, top_chunk: str, chunk_summaries) -> str:
    if top_chunk or chunk_summaries:
        research_context = f"Research Context: {top_chunk}\n{chunk_summaries}"
    else:
        research_context = ""

    prompt = f"""Create an article structure for: {topic}

{research_context}

Requirements:
Create exactly ONE article outline for topic
Plan exactly 6 meaningful sections with concise titles that logically cover the topic.

Respond with headings ONLY using this exact template (do not add explanations, bullet points, or extra text):

# {topic}
## [Short Title (max 8 words)]
## [Short Title (max 8 words)]
## [Short Title (max 8 words)]
## [Short Title (max 8 words)]
## [Short Title (max 8 words)]
## [Short Title (max 8 words)]

Rules:
Rules:
- Produce ONE outline only — do NOT repeat or restate it.
- Include exactly six (6) section titles, concise and ≤8 words.
- Do NOT add paragraphs, explanations, or duplicates.
- End output immediately after the sixth heading.
- After completing the outline, write the single word **[END]** on a new line.
"""
    return prompt


def select_section_chunks_prompt(
    section_heading: str, topic: str, chunk_summaries: str, num_chunks: int
) -> str:
    """Generate prompt for selecting relevant research chunks for a section."""
    return f"""
You are writing the section titled {section_heading} for the article {topic}.

Below are research chunk summaries:
{chunk_summaries}

Task:
Select only the chunk IDs that are most relevant for writing the {section_heading} section.

Strict Rules (must follow all):
- VALID IDS = exactly the literal tokens shown before each colon in the data (e.g., `search_supabase_faiss_0_d44af884`, `search_hybrid_6_201e0dd2`). Do NOT invent, shorten, number, or reformat IDs.
- AFL ONLY: Exclude chunks that are clearly about other sports, people, events, or unrelated topics.
- Return **no more than {num_chunks}** IDs that best explain what the {topic} is, its purpose, significance, and key context.
- Output format: IDs separated by commas, no spaces, no quotes, no extra text.
- End output immediately after the last ID.
- If nothing fits, respond exactly with `NONE`.

Validation Hints:
- ✅ Valid: `search_supabase_faiss_0_d44af884,search_hybrid_6_201e0dd2`
- ❌ Invalid: `6,7,8` or `"search_supabase_faiss_0_d44af884"` or `AFL_Grand_Final`

"""


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
    """Generate prompt for creating targeted search queries using entity extraction and relationship analysis."""
    context_section = ""
    if context:
        context_section = f"""
CONTEXT FROM INITIAL SEARCH:
{context}


First, extract key entities from this context:
- PEOPLE: Names of key individuals mentioned
- ORGANIZATIONS: Teams, companies, institutions
- LOCATIONS: Places, venues, geographic areas
- DATES/TIMES: Specific dates, periods, timeframes
- EVENTS: Specific incidents, competitions, ceremonies
- STATISTICS: Numbers, measurements, records
- CONCEPTS: Technical terms, processes, phenomena

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
- Include quantitative aspects (statistics, measurements, comparisons)
- Target controversial or disputed aspects when relevant

IMPORTANT: Output EXACTLY {num_queries} search queries. Each query should be a single line with 3-8 words maximum. No explanations, no numbering, no extra text.

FORMAT: Each line = one search query only

Example for "2022 World Cup Final":
Lionel Messi Kylian Mbappe individual performance statistics 2022 World Cup Final
Argentina France penalty shootout decision-making tactical analysis December 2022
Qatar World Cup Final attendance revenue compared previous World Cup finals
Lionel Scaloni Didier Deschamps tactical substitutions impact match outcome
2022 World Cup Final cultural impact Argentina national celebration aftermath
"""
