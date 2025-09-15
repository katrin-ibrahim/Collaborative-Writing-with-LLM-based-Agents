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
    return f"""
Create a simple outline for a comprehensive article about: {topic}.
You must output ONLY the outline headings.

Structure:
1. Title (as a markdown H1)
2. 4–6 main sections (as markdown H2 headings)
   - Section 1: introduction or background
   - Section 2: core concepts
   - Section 3: applications/examples
   - Section 4: current developments
   - Section 5: future/implications
   - Section 6: conclusion (optional)

Correct Format Example:
# Title: Example Topic

## Introduction & Context
## Core Concepts
## Applications
## Current Developments
## Future Implications
## Conclusion

Incorrect Format Example:
# Title: Example Topic
## Section 1
- Explanation text
## Section 2
- Bullet points
## Section 3 : subheading

Guidelines:
- Only output the section headings
- No bullets, no explanations, no subheadings
- Keep section titles concise
"""


def refinement_prompt(topic: str, current_outline: str, content_summary: str) -> str:
    """Generate prompt for outline refinement based on research."""
    return f"""
Refine this outline for "{topic}" based on research findings:

Current Outline:
{current_outline}

Content Summary from Research:
{content_summary}

Structure:
1. Title (as a markdown H1)
2. 4–6 main sections (as markdown H2 headings)
   - Section 1: introduction or background
   - Section 2: core concepts
   - Section 3: applications/examples
   - Section 4: current developments
   - Section 5: future/implications
   - Section 6: conclusion (optional)

Correct Format Example:
# Title: Example Topic

## Introduction & Context
## Core Concepts
## Applications
## Current Developments
## Future Implications
## Conclusion

Incorrect Format Example:
# Title: Example Topic
## Section 1
- Explanation text
## Section 2
- Bullet points
## Section 3 : subheading

Guidelines:
- Only output the section headings
- No bullets, no explanations, no subheadings
- Keep section titles concise
- Incorporate search insights to improve relevance and completeness of section headings
- There is no need to write section content at this stage, you are only using the research to enhance or create more targeted section headings



"""


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
