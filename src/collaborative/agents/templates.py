# ---------------------------------------- Reviwer Templates ---------------------------------------
# src/collaborative/agents/templates/reviewer_prompts.py
"""
Prompt template functions for ReviewerAgent workflow.
"""

from typing import Dict, List


def fact_checking_prompt(article_content: str, title: str) -> str:
    """Generate prompt to identify factual claims that need verification."""
    return f"""You are a fact-checking expert. Extract factual claims from this article about "{title}" that need verification.

ARTICLE CONTENT:
{article_content}

IMPORTANT: You must provide your response in the EXACT XML format shown below. Do not include <think> tags or explanations outside the XML.

Find 3-8 specific factual claims that can be verified (statistics, dates, names, events, quotes). Output them using this EXACT format:

<fact_check_analysis>
<claim id="1">
<text>exact claim text from the article</text>
<category>number</category>
<priority>high</priority>
<checkable>true</checkable>
</claim>
<claim id="2">
<text>another exact claim text</text>
<category>event</category>
<priority>medium</priority>
<checkable>true</checkable>
</claim>
</fact_check_analysis>

Categories: date, number, name, location, event, quote
Priorities: high, medium, low

Remember: Only output the XML format above. No additional text or thinking."""


def feedback_prompt(
    article_title: str,
    article_content: str,
    metrics: Dict,
    potential_claims: List[str],
    fact_check_results: List[Dict],
) -> str:
    """Generate informed feedback prompt based on metrics and fact-checking results."""

    # Format metrics for readability - handle both dict and ArticleMetrics object
    if hasattr(metrics, "word_count"):
        # ArticleMetrics object
        word_count = metrics.word_count
        heading_count = metrics.heading_count
        paragraph_count = metrics.paragraph_count
    else:
        # Dictionary
        word_count = metrics.get("word_count", 0)
        heading_count = metrics.get("heading_count", 0)
        paragraph_count = metrics.get("paragraph_count", 0)

    # Format claims
    claims_text = (
        "\n".join([f"- {claim}" for claim in potential_claims[:5]])
        if potential_claims
        else "No specific claims identified"
    )

    # Format fact-check results with verification status
    fact_check_text = ""
    if fact_check_results:
        for result in fact_check_results[:5]:
            claim = result.get("claim", "Unknown claim")
            sources_found = result.get("sources_found", 0)
            verified = "VERIFIED" if sources_found > 0 else "UNVERIFIED"
            fact_check_text += f"- '{claim}': {verified} ({sources_found} sources)\n"
    else:
        fact_check_text = "No fact-checking performed"

    return f"""Provide expert feedback for this article about "{article_title}" based on objective analysis and fact-checking results.

ARTICLE CONTENT:
{article_content[:1500]}{"..." if len(article_content) > 1500 else ""}

OBJECTIVE METRICS:
- Word Count: {word_count}
- Heading Count: {heading_count}
- Paragraph Count: {paragraph_count}

CLAIMS IDENTIFIED:
{claims_text}

FACT-CHECKING RESULTS:
{fact_check_text}

Provide structured feedback in this exact format:

## Content Quality Issues
- [Specific clarity or accuracy issues, especially for UNVERIFIED claims]
- [Problems with logical flow or organization]
- [Missing information or incomplete coverage]

## Structural Assessment
- [Evaluation of section organization and transitions]
- [Assessment of heading appropriateness and hierarchy]
- [Comments on overall article structure]

## Improvement Recommendations
- [Specific suggestions for content enhancement]
- [Recommendations for unverified claims - add sources or qualify statements]
- [Areas requiring additional research or development]

## Overall Assessment
[2-3 sentences providing comprehensive evaluation considering both content quality and factual accuracy]

Focus on factual accuracy issues revealed by the fact-checking results."""


# ---------------------------------------- Writer Templates ----------------------------------------
"""
Prompt template functions for WriterAgent workflow.
"""


def planning_prompt(topic: str) -> str:
    return f"""Create an article structure for: {topic}

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
- Each "##" line must be a brief title (≤ 8 words) with no sentences or punctuation beyond basic words.
- Do NOT include paragraphs, descriptions, or additional commentary.
- Ensure the six sections follow a logical narrative from introduction to conclusion.
- If you cannot comply, still output six placeholder headings in the required format.

Topic: {topic}"""


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


# ---------------------------------------- Reviewer Agentic Templates ---------------------------------------


def reviewer_tool_decision_prompt(
    article_title: str, available_tools: List[str]
) -> str:
    """Let reviewer agent decide which tools to use for comprehensive review."""
    return f"""You are reviewing an article titled "{article_title}".

AVAILABLE TOOLS: {available_tools}

DECISION TASK: Which tools should you use for a comprehensive review?

Consider:
1. Do you need to search for additional sources to verify claims?
2. Should you retrieve specific research chunks for fact-checking?
3. Do you need to check previous iterations for comparison?
4. Are there feedback tools that would improve your review?

Respond with:
REVIEW_TOOLS: tool1,tool2,tool3 (comma-separated)
REASONING: Explain why these tools will improve your review quality"""


def reviewer_search_strategy_prompt(article_title: str, claims: List[str]) -> str:
    """Reviewer decides what to search for to verify claims."""
    return f"""You are fact-checking claims from "{article_title}".

CLAIMS TO VERIFY:
{chr(10).join([f"- {claim}" for claim in claims[:5]])}

DECISION TASK: What should you search for to verify these claims?

Consider:
1. Which claims are most important to verify?
2. What search terms would find authoritative sources?
3. Should you do broad searches or specific targeted searches?

Respond with:
SEARCH_QUERIES: query1,query2,query3 (comma-separated)
PRIORITY_CLAIMS: claim numbers that are highest priority
REASONING: Explain your verification strategy"""


def reviewer_feedback_strategy_prompt(
    article_title: str,
    fact_check_results: List[Dict],
    available_context_tools: List[str],
) -> str:
    """Reviewer decides how to structure feedback using available context tools."""
    return f"""You are providing feedback for "{article_title}".

FACT-CHECK RESULTS:
- Total claims checked: {len(fact_check_results)}
- Claims with issues: {sum(1 for r in fact_check_results if not r.get('verified', True))}

AVAILABLE CONTEXT TOOLS: {available_context_tools}

DECISION TASK: How should you structure your feedback?

Consider:
1. Should you compare with previous iterations?
2. Do you need current iteration context?
3. Should you check existing feedback to avoid duplication?

Respond with:
CONTEXT_TOOLS: tool1,tool2 (or "NONE" if not needed)
FEEDBACK_FOCUS: areas to prioritize in feedback
REASONING: Explain your feedback strategy"""


def enhanced_feedback_prompt(base_feedback: str, supplementary_context: str) -> str:
    """Enhanced feedback prompt incorporating agentic tool context."""
    return f"""{base_feedback}

ADDITIONAL CONTEXT FROM AGENTIC TOOLS:
{supplementary_context}

ENHANCED INSTRUCTIONS:
- Incorporate insights from the additional context above
- Provide more comprehensive feedback based on the supplementary information
- If you found contradicting sources, note the discrepancies
- If iteration context is available, compare with previous versions

Focus on providing comprehensive feedback that utilizes all available context."""


def agentic_review_prompt(
    article_title: str,
    context_text: str,
    metrics: dict,
    fact_check_summary: str,
    article_content: str,
) -> str:
    """Context-aware agentic review prompt."""
    return f"""You are reviewing an article titled "{article_title}".

ITERATION & MEMORY CONTEXT:
{context_text}

ARTICLE METRICS:
- Word count: {metrics.word_count if hasattr(metrics, 'word_count') else metrics.get('word_count', 0)}
- Headings: {metrics.heading_count if hasattr(metrics, 'heading_count') else metrics.get('heading_count', 0)} ({', '.join((metrics.headings if hasattr(metrics, 'headings') else metrics.get('headings', []))[:5])})
- Paragraphs: {metrics.paragraph_count if hasattr(metrics, 'paragraph_count') else metrics.get('paragraph_count', 0)}

FACT-CHECK SUMMARY:
{fact_check_summary}

ARTICLE CONTENT:
{article_content}

TASK: Provide comprehensive feedback that considers the iteration context and fact-checking results.

Focus on improvement areas and specific recommendations based on the available context."""


def context_decision_prompt(
    topic: str, context_summary: str, outline_info: str, num_chunks: int
) -> str:
    """Prompt for deciding whether to select specific research chunks."""
    return f"""You are about to write an article about "{topic}".

CURRENT CONTEXT: {context_summary}
{outline_info}
AVAILABLE RESEARCH CHUNKS: {num_chunks} chunks

QUESTION: Do you need to select specific chunks before writing this article?

Respond with just "YES" if you need to select chunks, "NO" if you're ready to write with all available chunks."""


def chunk_selection_detailed_prompt(
    topic: str, outline_info: str, chunks_info: str
) -> str:
    """Detailed prompt for selecting specific research chunks."""
    return f"""You are writing about "{topic}".

{outline_info}

AVAILABLE RESEARCH CHUNKS:
{chunks_info}

TASK: Select which chunk IDs are most relevant for writing this article.

Respond with just the chunk IDs separated by commas (e.g., "search_direct_0,search_query_1") or "NONE" if nothing is relevant:"""


def writer_feedback_evaluation_prompt(
    topic: str, article_content: str, feedback_items: List[Dict]
) -> str:
    """
    Prompt for writer to evaluate reviewer feedback and decide whether to contest any items.
    This enables Theory of Mind research by allowing writer-reviewer disagreement.
    """
    feedback_text = ""
    for item in feedback_items:
        feedback_body = item.get("feedback") or item.get("text", "")
        target_section = item.get("target_section", "general")
        priority = item.get("priority", "medium")
        feedback_id = item.get("id", "unknown")

        feedback_text += f"""
FEEDBACK_ID: {feedback_id}
TARGET_SECTION: {target_section}
FEEDBACK: {feedback_body}
PRIORITY: {priority}

---"""

    return f"""You are the writer of an article about "{topic}". A reviewer has provided feedback on your work.

Your task is to evaluate each piece of feedback and decide whether to:
1. ACCEPT the feedback (you agree and will address it)
2. CONTEST the feedback (you disagree and want to challenge it)

CURRENT ARTICLE:
{article_content[:2000]}...

REVIEWER FEEDBACK TO EVALUATE:
{feedback_text}

For each piece of feedback, respond using this EXACT format:

FEEDBACK_ID: [feedback_id]
DECISION: ACCEPT or CONTEST
REASONING: [your reasoning for accepting or contesting this feedback]

Guidelines for contesting:
- Contest if the feedback is factually wrong
- Contest if it goes against the article's purpose or scope
- Contest if it asks for information that isn't relevant
- Contest if it contradicts well-established facts
- Accept if the feedback would genuinely improve the article
- Accept if it points out real errors or missing information

Be thoughtful but don't hesitate to contest feedback you genuinely disagree with."""


def reviewer_contest_response_prompt(
    original_feedback: str, writer_reasoning: str, article_excerpt: str
) -> str:
    """
    Prompt for reviewer to respond to writer's contested feedback.
    This enables negotiation and resolution of disagreements.
    """
    return f"""You are a reviewer. The writer has CONTESTED one of your feedback items, disagreeing with your suggestion.

YOUR ORIGINAL FEEDBACK:
{original_feedback}

WRITER'S CONTEST REASONING:
{writer_reasoning}

CURRENT ARTICLE EXCERPT:
{article_excerpt}...

As the reviewer, you need to evaluate the writer's contest and decide how to resolve this disagreement.

Your options:
1. ACCEPT_WRITER_POSITION - The writer is correct, withdraw your feedback
2. MAINTAIN_REVIEWER_POSITION - Your feedback is still valid, writer should address it
3. COMPROMISE - Find middle ground between your position and writer's position

Respond using this EXACT format:

RESOLUTION: [one of the three options above]
REASONING: [explain your decision and any compromise details]

Consider:
- Is the writer's reasoning factually correct?
- Does the writer have a valid point about scope or relevance?
- Is there a middle ground that addresses both concerns?
- Would your original feedback genuinely improve the article?"""
