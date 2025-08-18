from langchain_core.tools import tool
from typing import Any, Dict

from src.agents.writer.outline_generator import OutlineGenerator
from src.utils.data import Outline

# Global outline generator instance
_outline_generator = None


def _get_outline_generator(config: Dict[str, Any] = None):
    """Get or create outline generator instance."""
    global _outline_generator
    if _outline_generator is None:
        _outline_generator = OutlineGenerator(config or {})
    return _outline_generator


@tool
def generate_outline(topic: str, context: str = "") -> Dict[str, Any]:
    """
    Generate a structured outline for an article topic.

    Args:
        topic: The main topic for the article
        context: Optional context information to inform the outline

    Returns:
        Dictionary with outline structure and metadata for LLM processing
    """
    outline_generator = _get_outline_generator()
    outline = outline_generator.generate_outline(topic, context)

    return {
        "topic": topic,
        "title": outline.title,
        "headings": outline.headings,
        "subheadings": outline.subheadings,
        "sections_count": len(outline.headings),
        "summary": f"Generated outline for '{topic}' with {len(outline.headings)} main sections",
    }


@tool
def generate_section_content(
    section_title: str, context: str, article_topic: str, target_length: str = "300-400"
) -> Dict[str, Any]:
    """
    Generate content for a specific article section using provided context.

    Args:
        section_title: Title of the section to write
        context: Relevant information to base the content on
        article_topic: Main topic of the article
        target_length: Target word count range (default: "300-400")

    Returns:
        Dictionary with generation prompt for LLM processing
    """
    if context:
        prompt = f"""
        Write a comprehensive section titled "{section_title}" for an article about "{article_topic}".

        Use this relevant information:
        {context}

        Requirements:
        1. Write 2-3 well-structured paragraphs ({target_length} words)
        2. Ground your writing in the provided information
        3. Maintain an informative, engaging tone
        4. Include specific details and examples from the context
        5. Ensure logical flow and clear organization
        """
    else:
        prompt = f"""
        Write a comprehensive section titled "{section_title}" for an article about "{article_topic}".

        Requirements:
        1. Write 2-3 well-structured paragraphs ({target_length} words)
        2. Draw from your knowledge of {article_topic}
        3. Provide clear explanations and relevant examples
        4. Maintain an informative, engaging tone
        5. Ensure logical flow and organization
        """

    return {
        "section_title": section_title,
        "article_topic": article_topic,
        "generation_prompt": prompt,
        "summary": f"Prepared content generation for section '{section_title}'",
    }


class ContentToolkit:
    """Legacy wrapper for backward compatibility."""

    def __init__(self, config: Dict[str, Any]):
        global _outline_generator
        _outline_generator = OutlineGenerator(config)

    def generate_outline(self, topic: str, context: str = "") -> Outline:
        """Legacy method - use generate_outline tool instead."""
        result = generate_outline.invoke({"topic": topic, "context": context})
        return Outline(
            title=result["title"],
            headings=result["headings"],
            subheadings=result["subheadings"],
        )

    def generate_section_content(
        self,
        section_title: str,
        context: str,
        article_topic: str,
        api_client,
        target_length: str = "300-400",
    ) -> str:
        """Legacy method - use generate_section_content tool and LLM call."""
        result = generate_section_content.invoke(
            {
                "section_title": section_title,
                "context": context,
                "article_topic": article_topic,
                "target_length": target_length,
            }
        )
        return api_client.call_api(result["generation_prompt"])
