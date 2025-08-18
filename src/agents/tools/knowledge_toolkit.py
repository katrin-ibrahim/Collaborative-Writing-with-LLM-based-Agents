from langchain_core.tools import tool
from typing import Any, Dict, List

from src.knowledge.knowledge_base import KnowledgeBase
from src.utils.data import SearchResult


@tool
def create_knowledge_organizer(topic: str) -> Dict[str, Any]:
    """
    Create a new knowledge organizer for a specific topic.

    Args:
        topic: The main topic to organize knowledge around

    Returns:
        Dictionary with organizer metadata and initialization status
    """
    organizer = KnowledgeBase(topic)

    return {
        "topic": topic,
        "organizer_id": id(organizer),  # Simple ID for tracking
        "status": "initialized",
        "categories_count": 0,
        "results_count": 0,
        "summary": f"Created knowledge organizer for topic: '{topic}'",
    }


@tool
def organize_for_writing(
    topic: str,
    categories: List[Dict[str, str]],
    search_results_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Organize search results into categories optimized for content creation.

    Args:
        topic: The main topic being organized
        categories: List of category dictionaries with 'name' and 'description'
        search_results_data: List of search result dictionaries

    Returns:
        Dictionary with organization results and category assignments
    """
    organizer = KnowledgeBase(topic)
    search_results = [SearchResult(**data) for data in search_results_data]

    # Create categories
    category_nodes = {}
    for category in categories:
        node = organizer.create_category(
            category["name"], category.get("description", "")
        )
        category_nodes[category["name"]] = node

    # Organize results by categories
    category_assignments = {}
    for category_name, node in category_nodes.items():
        similar_results = organizer.find_most_similar_results(
            category_name + " " + node.content,
            search_results,
            top_k=max(1, len(search_results) // len(categories)),
        )
        organizer.assign_results_to_category(similar_results, node)
        category_assignments[category_name] = len(similar_results)

    return {
        "topic": topic,
        "categories_created": len(categories),
        "total_results": len(search_results),
        "category_assignments": category_assignments,
        "organizer_id": id(organizer),
        "summary": f"Organized {len(search_results)} results into {len(categories)} categories for writing",
    }


@tool
def organize_for_verification(
    topic: str, claims_to_verify: List[str], search_results_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Organize search results for fact-checking and verification purposes.

    Args:
        topic: The main topic being verified
        claims_to_verify: List of claims that need verification
        search_results_data: List of search result dictionaries

    Returns:
        Dictionary with verification-focused organization results
    """
    organizer = KnowledgeBase(topic)
    search_results = [SearchResult(**data) for data in search_results_data]

    verification_categories = [
        {
            "name": "Factual Claims",
            "description": "Verifiable facts and statistics",
        },
        {
            "name": "Source Evidence",
            "description": "Supporting evidence from reliable sources",
        },
        {
            "name": "Contradictions",
            "description": "Conflicting information that needs resolution",
        },
        {
            "name": "Uncertain Claims",
            "description": "Claims that need additional verification",
        },
    ]

    # Create verification categories
    category_nodes = {}
    for category in verification_categories:
        node = organizer.create_category(
            category["name"], category.get("description", "")
        )
        category_nodes[category["name"]] = node

    # Organize results for verification
    verification_assignments = {}
    for category_name, node in category_nodes.items():
        similar_results = organizer.find_most_similar_results(
            category_name + " " + node.content,
            search_results,
            top_k=max(1, len(search_results) // len(verification_categories)),
        )
        organizer.assign_results_to_category(similar_results, node)
        verification_assignments[category_name] = len(similar_results)

    return {
        "topic": topic,
        "claims_to_verify": len(claims_to_verify),
        "verification_categories": len(verification_categories),
        "total_results": len(search_results),
        "verification_assignments": verification_assignments,
        "organizer_id": id(organizer),
        "summary": f"Organized {len(search_results)} results for verifying {len(claims_to_verify)} claims",
    }


@tool
def find_relevant_content(
    topic: str,
    target_description: str,
    search_results_data: List[Dict[str, Any]],
    max_results: int = 5,
) -> Dict[str, Any]:
    """
    Find content most relevant to a target description from available results.

    Args:
        topic: The main topic context
        target_description: Description of what content to find
        search_results_data: List of available search result dictionaries
        max_results: Maximum number of relevant results to return

    Returns:
        Dictionary with most relevant results and relevance scores
    """
    organizer = KnowledgeBase(topic)
    search_results = [SearchResult(**data) for data in search_results_data]

    relevant_results = organizer.find_most_similar_results(
        target_description, search_results, max_results
    )

    return {
        "topic": topic,
        "target_description": target_description,
        "total_available": len(search_results),
        "relevant_found": len(relevant_results),
        "relevant_results": [result.to_dict() for result in relevant_results],
        "summary": f"Found {len(relevant_results)} relevant results for '{target_description}'",
    }


@tool
def extract_claims_from_content(content: str) -> Dict[str, Any]:
    """
    Extract verifiable claims from text content for fact-checking.

    Args:
        content: Text content to analyze for claims

    Returns:
        Dictionary with extracted claims and analysis metadata
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", content)

    # Simple heuristic for finding factual claims
    claims = []
    for sentence in sentences:
        if (
            any(
                word in sentence.lower()
                for word in ["is", "are", "was", "were", "has", "have"]
            )
            and len(sentence.split()) > 5
            and not sentence.strip().startswith(
                ("However", "Moreover", "Furthermore", "In conclusion")
            )
        ):
            claims.append(sentence.strip())

    # Limit for practical use
    extracted_claims = claims[:10]

    return {
        "content_length": len(content),
        "total_sentences": len(sentences),
        "claims_found": len(extracted_claims),
        "extracted_claims": extracted_claims,
        "analysis_method": "heuristic_pattern_matching",
        "summary": f"Extracted {len(extracted_claims)} verifiable claims from {len(sentences)} sentences",
    }


class KnowledgeToolkit:
    """
    Legacy wrapper for backward compatibility.
    New implementations should use the @tool decorated functions directly.
    """

    def create_organizer(self, topic: str) -> KnowledgeBase:
        """Legacy method - use create_knowledge_organizer tool instead."""
        return KnowledgeBase(topic)

    def organize_for_writing(
        self,
        organizer: KnowledgeBase,
        categories: List[Dict[str, str]],
        search_results: List[SearchResult],
    ) -> None:
        """Legacy method - use organize_for_writing tool instead."""
        self._organize_by_categories(organizer, categories, search_results)

    def organize_for_verification(
        self,
        organizer: KnowledgeBase,
        claims_to_verify: List[str],
        search_results: List[SearchResult],
    ) -> None:
        """Legacy method - use organize_for_verification tool instead."""
        verification_categories = [
            {
                "name": "Factual Claims",
                "description": "Verifiable facts and statistics",
            },
            {
                "name": "Source Evidence",
                "description": "Supporting evidence from reliable sources",
            },
            {
                "name": "Contradictions",
                "description": "Conflicting information that needs resolution",
            },
            {
                "name": "Uncertain Claims",
                "description": "Claims that need additional verification",
            },
        ]
        self._organize_by_categories(organizer, verification_categories, search_results)

    def _organize_by_categories(
        self,
        organizer: KnowledgeBase,
        categories: List[Dict[str, str]],
        search_results: List[SearchResult],
    ) -> None:
        """Internal method for category-based organization."""
        category_nodes = {}
        for category in categories:
            node = organizer.create_category(
                category["name"], category.get("description", "")
            )
            category_nodes[category["name"]] = node

        for category_name, node in category_nodes.items():
            similar_results = organizer.find_most_similar_results(
                category_name + " " + node.content,
                search_results,
                top_k=max(1, len(search_results) // len(categories)),
            )
            organizer.assign_results_to_category(similar_results, node)

    def find_relevant_content(
        self, organizer: KnowledgeBase, target_description: str, max_results: int = 5
    ) -> List[SearchResult]:
        """Legacy method - use find_relevant_content tool instead."""
        all_results = organizer.get_all_search_results()
        return organizer.find_most_similar_results(
            target_description, all_results, max_results
        )

    def extract_claims_from_content(self, content: str) -> List[str]:
        """Legacy method - use extract_claims_from_content tool instead."""
        result = extract_claims_from_content.invoke({"content": content})
        return result["extracted_claims"]
