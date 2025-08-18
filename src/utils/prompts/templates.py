"""
Prompt templates for baseline experiment methods.
Centralizes all prompt generation logic for consistency across methods.
"""


def build_direct_prompt(topic: str) -> str:
    """Build direct prompting prompt (shared between Ollama and local baselines)."""
    return f"""Write a comprehensive, well-structured Wikipedia-style article about \"{topic}\".

You are an expert encyclopedia writer. Create a detailed, factual article that captures the essential information about this topic.

CRITICAL REQUIREMENTS:
1. Start with a strong, informative introduction that summarizes the key facts
2. Create 4-6 main sections with specific, descriptive headings (NOT generic ones like "Overview")
3. Each section should contain 2-3 substantial paragraphs with specific details
4. Include dates, numbers, names, and concrete facts wherever possible
5. Use proper Wikipedia-style citations format [1], [2], etc. (even if hypothetical)
6. Maintain an encyclopedic, neutral tone throughout
7. Target 1200-1600 words for comprehensive coverage
8. Include entity-rich content with proper nouns, technical terms, and specific details

SECTION STRATEGY:
- Choose section headings that are specific to the topic domain
- For events: Background, Timeline, Key figures, Impact, Aftermath
- For organizations: History, Structure, Operations, Services, Controversies
- For people: Early life, Career, Major achievements, Legacy
- For places: Geography, History, Demographics, Economy, Culture
- For concepts: Definition, Development, Applications, Criticism

FORMAT:
# {topic}

[Write a comprehensive 2-3 paragraph introduction that defines the topic, explains its significance, and provides key contextual information. Include specific dates, locations, and quantitative details.]

## [Section 1 - Specific heading related to topic]

[2-3 detailed paragraphs with specific facts, dates, names, and quantitative information. Include proper citations.]

## [Section 2 - Another specific heading]

[2-3 detailed paragraphs continuing the comprehensive coverage.]

## [Section 3 - Third specific heading]

[2-3 detailed paragraphs with continued depth and specificity.]

## [Section 4 - Fourth specific heading]

[2-3 detailed paragraphs maintaining encyclopedic quality.]

## [Section 5 - Fifth specific heading if needed]

[2-3 detailed paragraphs for comprehensive coverage.]

## [Section 6 - Final specific heading if needed]

[2-3 detailed paragraphs completing the comprehensive article.]

Write the complete article now."""


def build_rag_prompt(topic: str, context: str) -> str:
    """Build RAG prompt with retrieved context (shared utility)."""
    return f"""Write a comprehensive Wikipedia-style article about "{topic}" using the provided context.

Context Information:
{context}

Guidelines:
- Use the context to write an accurate, well-structured article
- Organize information into clear sections
- Write in encyclopedic style
- Focus on factual, verifiable information
- Create a comprehensive overview of the topic

Write the article:"""


def enhance_content_prompt(topic: str, content: str) -> str:
    """Build enhancement prompt for short content (shared utility)."""
    return f"""The following article about "{topic}" needs to be enhanced and expanded to meet Wikipedia standards.

Current article:
{content}

Please rewrite and expand this article to be comprehensive, well-structured, and informative. Focus on:
- Adding missing important information
- Improving organization and flow
- Ensuring factual accuracy
- Using encyclopedic tone
- Creating proper sections and subsections

Write the enhanced article:"""
