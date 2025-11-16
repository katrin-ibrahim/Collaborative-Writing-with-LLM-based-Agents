"""
Text processing utilities for article generation.
"""

import re


def remove_citation_tags(text: str) -> str:
    """
    Remove citation tags from text before saving final article.

    Citation tags like <c cite="hash"/> are useful during writing/review
    but should be removed before evaluation to avoid confusing the LLM judge.

    Args:
        text: Text potentially containing citation tags

    Returns:
        Text with all citation tags removed

    Examples:
        >>> remove_citation_tags("Hello <c cite=\"123\"/> world")
        'Hello world'
        >>> remove_citation_tags("Test <c cite=\"abc\"/>.")
        'Test.'
    """
    # Remove <c cite="..."/> tags
    cleaned = re.sub(r'<c\s+cite="[^"]+"\s*/>', "", text)
    cleaned = re.sub(r"<needs_source\s*/>", "", cleaned)
    # Remove any other citation tag variants
    cleaned = re.sub(r"<c[^>]*>", "", cleaned)
    cleaned = re.sub(r"</c>", "", cleaned)
    # Clean up any multiple spaces (but preserve newlines)
    cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
    # Clean up space before punctuation
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    return cleaned.strip()
