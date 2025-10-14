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
    5. Maintain an encyclopedic, neutral tone throughout
    6. Target 1600-2000 words for comprehensive coverage
    7. Include entity-rich content with proper nouns, technical terms, and specific details

    SECTION STRATEGY:
    - Choose section headings that are specific to the topic domain
    - For events: Background, Timeline, Key figures, Impact, Aftermath
    - For organizations: History, Structure, Operations, Services, Controversies
    - For people: Early life, Career, Major achievements, Legacy
    - For places: Geography, History, Demographics, Economy, Culture
    - For concepts: Definition, Development, Applications, Criticism


    You must write from general knowledge WITHOUT any citations, references, URLs, or source attributions whatsoever.

    FORMAT:
    # {topic}

    [Write a comprehensive 2-3 paragraph introduction that defines the topic, explains its significance, and provides key contextual information. Include specific dates, locations, and quantitative details but NO citations.]

    ## [Section 1 - Specific heading related to topic]

    [2-3 detailed paragraphs with specific facts, dates, names, and quantitative information. NO citations allowed.]

    ## [Section 2 - Another specific heading]

    [2-3 detailed paragraphs continuing the comprehensive coverage. NO citations allowed.]

    ## [Section 3 - Third specific heading]

    [2-3 detailed paragraphs with continued depth and specificity.]

    ## [Section 4 - Fourth specific heading]

    [2-3 detailed paragraphs maintaining encyclopedic quality.]

    ## [Section 5 - Fifth specific heading if needed]

    [2-3 detailed paragraphs for comprehensive coverage. NO citations.]

    ## [Section 6 - Final specific heading if needed]

    [2-3 detailed paragraphs completing the comprehensive article. NO citations.]

    Write the complete article NOW with ZERO citations or references."""


def build_query_generator_prompt(topic: str, num_queries: int = 5) -> str:
    return f"""Generate {num_queries} possible Wikipedia article titles related to the topic "{topic}".

                Guidelines:
                - Use your internal knowledge and understanding of the topic to analyze the given topic and generate an internal outline for the Wikipedia article you are writing.
                - Ask yourself:
                    What do I know about this topic?
                    What do I think this topic article could contain?
                    What would I search for if I were preparing to write a full Wikipedia article?
                    Am I considering the most relevant and specific aspects of the topic?
                    Am I certain that I have covered all the important aspects of the topic?
                    Am I confident that these titles accurately reflect the content of the article?
                - Then generate an internal outline for a for a Wikipedia article.
                - Using your internal outline, generate a list of possible article titles that could be used for a Wikipedia article.
                - Each title should be a concise, descriptive phrase that accurately reflects the content of the article.
                - Titles should be suitable for a Wikipedia article, avoiding overly broad or generic titles.
                - Each title should be a single line, without numbering or extra text, and should be distinct from other titles.
                - Avoid using phrases that could lead to irrelevant results.
                - Avoid phrasing like questions, search queries, or casual writing.


                CRITICAl INSTRUCTION: ONLY output possible Wikipedia article titles — one per line, no numbering and absolutely NO extra text.

                Wikipedia article titles for "{topic}"
                """


def build_rag_prompt(topic: str, context: str) -> str:
    return f"""
    SYSTEM INSTRUCTION (do not ignore):

    You are writing a *Wikipedia-style* article using the provided context.
    Some of the context may be irrelevant or low-quality, so you must carefully select and use only the most relevant information.

    ===================================================
    == MANDATORY STRUCTURE (must match exactly) ==
    ===================================================

    # {topic}

    ## Introduction
    [2–3 paragraphs defining the topic with inline citations like [1], [2]]

    ## Section 1 - [Descriptive Heading]
    [Detailed paragraphs with inline citations like [3], [4]]

    ## Section 2 - [Descriptive Heading]
    [Detailed paragraphs continuing coverage with citations like [5], [6]]

    ## References
    1. [chunk_id] Title - URL
    2. [chunk_id] Title (Source: source_name)

    ===================================================
    == MANDATORY CITATION RULES (STRICT) ==
    ===================================================

    1. Every paragraph must contain at least one numbered inline citation in the form [N].
    2. Each [N] must correspond to a valid chunk_id present in the context below.
    3. NEVER invent citations or URLs.
    4. If a statement cannot be supported by context, omit it.
    5. The '## References' section must enumerate all used citations exactly as above.

    ===================================================
    == CONTEXT INFORMATION ==
    ===================================================
    {context}

    ===================================================
    BEGIN ARTICLE OUTPUT BELOW — FOLLOW FORMAT EXACTLY
    ===================================================
    """
