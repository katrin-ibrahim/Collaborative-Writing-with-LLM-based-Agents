# src/knowledge/knowledge_base.py
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional

from utils.data_models import SearchResult


@dataclass
class KnowledgeNode:
    """
    A hierarchical node for sophisticated knowledge organization.

    This class represents a single concept or piece of information within
    a larger knowledge structure. It can contain both content and relationships
    to other concepts.


    """

    title: str = ""
    content: str = ""
    source: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["KnowledgeNode"] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    semantic_embedding: Optional[np.ndarray] = None

    def add_search_result(self, search_result: SearchResult):
        """Add a search result to this knowledge node."""
        self.search_results.append(search_result)

        if not self.content:
            self.content = search_result.content
            self.source = search_result.source

    def add_child(self, child: "KnowledgeNode") -> "KnowledgeNode":
        """Add a child node, creating hierarchical relationships."""
        self.children.append(child)
        return child

    def get_all_content(self) -> str:
        """Get all content associated with this node and its children."""
        content_parts = []

        if self.content:
            content_parts.append(self.content)

        for snippet in self.snippets:
            if "content" in snippet:
                content_parts.append(snippet["content"])

        for child in self.children:
            child_content = child.get_all_content()
            if child_content:
                content_parts.append(child_content)

        return "\n\n".join(content_parts)


class KnowledgeBase:
    """
      Pure knowledge organization system.

    Responsibilities:
    1. Store SearchResult objects in hierarchical structure
    2. Provide semantic similarity calculations
    3. Retrieve content by categories/nodes

    """

    def __init__(self, topic: str):
        self.topic = topic
        self.root = KnowledgeNode(title=topic)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"KnowledgeBase initialized for topic: '{topic}'")

    def add_search_results(self, search_results: List[SearchResult]) -> None:
        """Add search results to root node."""
        for result in search_results:
            self.root.add_search_result(result)

    def create_category(
        self, category_name: str, category_description: str = ""
    ) -> KnowledgeNode:
        """Create a new category node and return it."""
        category_node = KnowledgeNode(title=category_name, content=category_description)
        self.root.add_child(category_node)
        return category_node

    def assign_results_to_category(
        self, search_results: List[SearchResult], category_node: KnowledgeNode
    ) -> None:
        """Assign specific search results to a category."""
        for result in search_results:
            category_node.add_search_result(result)

    def find_most_similar_results(
        self, target_text: str, search_results: List[SearchResult], top_k: int = 5
    ) -> List[SearchResult]:
        """Find search results most similar to target text."""
        if not search_results:
            return []

        target_embedding = self.embedder.encode(target_text)
        similarities = []

        for result in search_results:
            result_embedding = self.embedder.encode(result.content)
            similarity = np.dot(target_embedding, result_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(result_embedding)
            )
            similarities.append((result, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [result for result, _ in similarities[:top_k]]

    def get_content_by_category(self, category_name: str) -> str:
        """Get all content from a specific category."""
        for category_node in self.root.children:
            if category_node.title.lower() == category_name.lower():
                return category_node.get_all_content()
        return ""

    def get_all_search_results(self) -> List[SearchResult]:
        """Get all search results stored in the organizer."""
        all_results = []

        def collect_results(node: KnowledgeNode):
            all_results.extend(node.search_results)
            for child in node.children:
                collect_results(child)

        collect_results(self.root)
        return all_results

    def get_categories(self) -> List[str]:
        """Get list of all category names."""
        return [child.title for child in self.root.children]

    def get_all_sources(self) -> List[str]:
        """Get list of all information sources."""
        sources = set()

        def collect_sources(node: KnowledgeNode):
            for result in node.search_results:
                sources.add(result.source)
            for child in node.children:
                collect_sources(child)

        collect_sources(self.root)
        return list(sources)

    def get_content_summary(self) -> Dict[str, Any]:
        """Get summary of organized content for agent decision-making."""
        return {
            "topic": self.topic,
            "total_search_results": len(self.get_all_search_results()),
            "categories": [
                {
                    "name": child.title,
                    "result_count": len(child.search_results),
                    "has_content": bool(child.content),
                }
                for child in self.root.children
            ],
            "all_sources": self.get_all_sources(),
        }
