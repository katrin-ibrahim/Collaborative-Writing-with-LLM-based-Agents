# ---------------------------------------- Reviwer Templates ---------------------------------------
# src/collaborative/agents/templates/reviewer_prompts.py
"""
Prompt template functions for ReviewerAgent workflow.
"""

from typing import Dict, List


def qualitative_feedback_prompt(
    article_title: str,
    article_content: str,
    metrics: Dict,
    potential_claims: List[str],
    fact_check_results: List[Dict],
) -> str:
    """Generate prompt for qualitative feedback based on metrics and fact-checking."""

    # Format metrics for readability
    word_count = metrics.get("word_count", 0)
    heading_count = metrics.get("heading_count", 0)
    paragraph_count = metrics.get("paragraph_count", 0)

    # Format claims
    claims_text = (
        "\n".join([f"- {claim}" for claim in potential_claims[:5]])
        if potential_claims
        else "No specific claims identified"
    )

    # Format fact-check results
    fact_check_text = ""
    if fact_check_results:
        for result in fact_check_results[:3]:
            claim = result.get("claim", "Unknown claim")
            sources_found = result.get("sources_found", 0)
            fact_check_text += (
                f"- '{claim}': {sources_found} sources found for verification\n"
            )
    else:
        fact_check_text = "No fact-checking performed"

    return f"""
Provide qualitative feedback for this article. Objective metrics have been calculated separately.

ARTICLE: {article_title}

CONTENT:
{article_content[:1500]}{"..." if len(article_content) > 1500 else ""}

OBJECTIVE METRICS:
- Word Count: {word_count}
- Heading Count: {heading_count}
- Paragraph Count: {paragraph_count}

POTENTIAL CLAIMS IDENTIFIED:
{claims_text}

FACT-CHECKING RESULTS:
{fact_check_text}

Provide qualitative feedback in this format:

MAIN ISSUES:
- [List 2-3 specific issues you observe]
- [Focus on clarity, coherence, completeness]
- [Be concrete and actionable]

RECOMMENDATIONS:
- [Provide 2-4 specific improvement suggestions]
- [Focus on content quality and structure]
- [Consider the metrics and fact-checking results above]

DETAILED FEEDBACK:
[2-3 sentences with your overall qualitative assessment]

Focus on:
- Content clarity and logical flow
- Completeness of coverage for the topic
- Writing quality and readability
- Structural organization
- Accuracy based on fact-checking results

Do NOT provide numerical scores - these have been calculated objectively.
"""


def fact_checking_strategy_prompt(article_content: str, title: str) -> str:
    """Generate prompt to identify claims that need fact-checking."""
    return f"""
    Extract factual claims that need verification.

    CONTENT:
    {article_content}

    Use this EXACT format (choose ONE value for each field):

    <fact_check_analysis>
    <claim id="1">
    <text>exact claim text here</text>
    <category>event</category>
    <priority>high</priority>
    <checkable>true</checkable>
    </claim>
    <claim id="2">
    <text>another claim</text>
    <category>number</category>
    <priority>medium</priority>
    <checkable>true</checkable>
    </claim>
    </fact_check_analysis>

    Categories: date, number, name, location, event, quote
    Priorities: high, medium, low
    Checkable: true, false

    Extract 5-10 claims minimum."""


def structure_analysis_prompt(article_content: str, title: str, metrics: Dict) -> str:
    """Generate prompt for analyzing article structure beyond basic metrics."""

    word_count = metrics.get("word_count", 0)
    heading_count = metrics.get("heading_count", 0)
    headings = metrics.get("headings", [])

    headings_text = (
        "\n".join([f"- {heading}" for heading in headings])
        if headings
        else "No headings found"
    )

    return f"""
Analyze the structure and organization of this article about "{title}":

BASIC METRICS:
- Word Count: {word_count}
- Heading Count: {heading_count}

HEADINGS FOUND:
{headings_text}

ARTICLE CONTENT:
{article_content[:1000]}{"..." if len(article_content) > 1000 else ""}

Evaluate the structural quality:

STRUCTURAL ASSESSMENT:
- Is the article well-organized with logical flow?
- Are the headings appropriate and comprehensive?
- Does each section contribute meaningfully to the topic?
- Is the length appropriate for the topic complexity?

SPECIFIC ISSUES:
- [List any structural problems you identify]
- [Note missing sections or organizational issues]

STRUCTURAL RECOMMENDATIONS:
- [Suggest specific improvements to organization]
- [Recommend additional sections if needed]

Focus on how well the structure serves the reader's understanding of the topic.
"""


# ---------------------------------------- Writer Templates ----------------------------------------
"""
Prompt template functions for WriterAgent workflow.
"""


def planning_prompt(topic: str) -> str:
    return f"""Create exactly 5 sections for an article about: {topic}

Output ONLY this format:
# Article Title
## Section 1
## Section 2
## Section 3
## Section 4
## Section 5

No explanations. No bullets. No additional text.

Topic: {topic}

Output:"""


def section_content_prompt_with_research(
    section_heading: str, topic: str, relevant_info: str
) -> str:
    """Generate prompt for section content with research information."""
    return f"""
Write a comprehensive section titled "{section_heading}" for an article about "{topic}".

Use this relevant information from research:
{relevant_info}

Requirements:
- Write 2-3 well-structured paragraphs
- Ground your content in the provided research information
- Include specific details and examples from the sources
- Maintain an informative, engaging tone
- Ensure smooth transitions and logical flow
- Avoid repetition of information from other sections
"""


def section_content_prompt_without_research(section_heading: str, topic: str) -> str:
    """Generate prompt for section content without research information."""
    return f"""
Write a comprehensive section titled "{section_heading}" for an article about "{topic}".

Requirements:
- Write 2-3 well-structured paragraphs
- Draw from your knowledge of {topic}
- Provide clear explanations and relevant examples
- Maintain an informative, engaging tone
- Ensure logical flow and organization
"""


def improvement_prompt(
    topic: str, article_content: str, feedback_text: str, recommendations: str
) -> str:
    """Generate prompt for improving article based on feedback."""
    return f"""
Improve the following article about "{topic}" based on this feedback:

Current Article:
{article_content}

Reviewer Feedback:
{feedback_text}

Specific Issues to Address:
{recommendations}

Instructions:
1. Address the feedback while maintaining the article's structure
2. Improve clarity, accuracy, and completeness
3. Add missing information where identified
4. Maintain consistent tone and style
5. Ensure smooth flow between sections

Provide the improved article:
"""


def search_query_generation_prompt(
    topic: str, context: str = "", num_queries: int = 5
) -> str:
    """Generate prompt for creating targeted search queries using entity extraction and relationship analysis."""
    context_section = ""
    if context.strip():
        context_section = f"""
CONTEXT FROM INITIAL SEARCH:
{context[:1200]}{"..." if len(context) > 1200 else ""}

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
