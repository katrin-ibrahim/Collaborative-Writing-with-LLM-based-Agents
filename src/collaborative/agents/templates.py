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
Analyze this article about "{title}" and identify the most important factual claims that should be verified:

ARTICLE CONTENT:
{article_content[:1000]}{"..." if len(article_content) > 1000 else ""}

Please identify:
1. Statistical claims (numbers, percentages, dates)
2. Factual assertions that could be verified
3. Claims about recent events or developments
4. Technical or scientific statements

Return the top 5 most important claims to fact-check, formatted as:
1. [First claim to verify]
2. [Second claim to verify]
3. [Third claim to verify]
4. [Fourth claim to verify]
5. [Fifth claim to verify]

Focus on claims that are:
- Specific and verifiable
- Important to the article's credibility
- Potentially controversial or disputed
"""


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
    """Generate prompt for initial outline planning."""
    return f"""
Create a detailed outline for a comprehensive article about: {topic}

Consider:
- What are the essential aspects readers need to understand?
- What background information is required?
- What are the key concepts, applications, or examples?
- What current developments or future implications exist?
- How should information be logically organized?

Create an outline with:
1. A clear, specific title
2. 4-6 main sections that thoroughly cover the topic
3. Logical flow from basic concepts to advanced topics

Format as:
Title: [Specific Title]

1. [Section 1 - Usually introduction/background]
2. [Section 2 - Core concepts]
3. [Section 3 - Applications/examples]
4. [Section 4 - Current state/developments]
5. [Section 5 - Future/implications]
6. [Section 6 - Conclusion (if needed)]
"""


def refinement_prompt(
    topic: str, current_outline: str, knowledge_summary: str, coverage_analysis: str
) -> str:
    """Generate prompt for outline refinement based on research."""
    return f"""
Refine this outline for "{topic}" based on research findings:

Current Outline:
{current_outline}

Available Knowledge Categories:
{knowledge_summary}

Knowledge Coverage Analysis:
{coverage_analysis}

Instructions:
1. Keep well-supported sections with sufficient information
2. Modify or merge sections with insufficient coverage
3. Add new sections for important topics discovered in research
4. Ensure logical flow and comprehensive coverage
5. Aim for 4-6 main sections maximum

Provide the refined outline in the same format as the original.
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


def knowledge_organization_prompt(topic: str, search_results: str) -> str:
    """Generate prompt for organizing search results."""
    return f"""
Organize the following search results for writing an article about "{topic}":

Search Results:
{search_results}

Please organize this information into logical categories that would be useful for writing.
For each category, list the relevant information and indicate how many sources support each point.

Format as:
Category 1: [Name]
- [Key point 1 from sources]
- [Key point 2 from sources]

Category 2: [Name]
- [Key point 1 from sources]
- [Key point 2 from sources]

Provide a brief summary of coverage for each category.
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
