from typing import Any, Dict, List

from agents.writer.outline_generator import OutlineGenerator
from utils.data_models import Outline, SearchResult


class ContentToolkit:
    """
    Content generation tools specific to writing tasks.

    These tools are primarily for writer agents, though reviewers might
    use some for generating feedback or suggestions.
    """

    def __init__(self, config: Dict[str, Any]):
        self.outline_generator = OutlineGenerator(config)

    def generate_outline(self, topic: str, context: str = "") -> Outline:
        """Generate an outline for the topic."""
        return self.outline_generator.generate_outline(topic, context)

    def create_context_from_results(
        self, search_results: List[SearchResult], max_length: int = 2000
    ) -> str:
        """Create context string from search results."""
        context_parts = []
        current_length = 0

        for i, result in enumerate(search_results):
            if current_length >= max_length:
                break

            snippet = f"[Source {i+1}]: {result.content}"
            if current_length + len(snippet) <= max_length:
                context_parts.append(snippet)
                current_length += len(snippet)
            else:
                remaining = max_length - current_length
                context_parts.append(snippet[:remaining] + "...")
                break

        return "\n\n".join(context_parts)

    def generate_section_content(
        self,
        section_title: str,
        context: str,
        article_topic: str,
        api_client,
        target_length: str = "300-400",
    ) -> str:
        """Generate content for a specific section."""
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

        return api_client.call_api(prompt)
