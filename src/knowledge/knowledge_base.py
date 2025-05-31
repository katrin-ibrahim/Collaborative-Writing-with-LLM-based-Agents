# src/knowledge/knowledge_base.py - Enhanced and restructured version
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

# Import our SearchResult type for integration with SearchEngine
from ..utils.data_models import SearchResult




@dataclass
class KnowledgeNode:
    """
    A hierarchical node for sophisticated knowledge organization.
    
    This class represents a single concept or piece of information within
    a larger knowledge structure. It can contain both content and relationships
    to other concepts, enabling rich knowledge representation.
    
    Think of this like a node in a mind map - it has its own content but also
    connects to related ideas in meaningful ways.
    """
    title: str = ""
    content: str = ""
    source: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["KnowledgeNode"] = field(default_factory=list)
    snippets: List[Dict[str, Any]] = field(default_factory=list)
    semantic_embedding: Optional[np.ndarray] = None

    def add_snippet(self, snippet: Dict[str, Any]):
        """
        Add a content snippet to this knowledge node.
        
        This method demonstrates how to incrementally build knowledge
        representations by adding related pieces of information.
        """
        self.snippets.append(snippet)
        
        # Update content if this node doesn't have content yet
        # This prioritizes the first meaningful content we find
        if not self.content and "content" in snippet:
            self.content = snippet["content"]
            
        # Update metadata with snippet information
        if "source" in snippet:
            if "sources" not in self.metadata:
                self.metadata["sources"] = []
            self.metadata["sources"].append(snippet["source"])

    def add_child(self, child: "KnowledgeNode") -> "KnowledgeNode":
        """Add a child node, creating hierarchical relationships."""
        self.children.append(child)
        return child

    def calculate_semantic_embedding(self, embedder: SentenceTransformer):
        """
        Calculate semantic embedding for this node's content.
        
        This enables intelligent content organization based on semantic
        similarity rather than just keyword matching.
        """
        if self.content:
            self.semantic_embedding = embedder.encode(self.content)
        elif self.title:
            self.semantic_embedding = embedder.encode(self.title)

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
    Sophisticated knowledge organization system.
    
    This class transforms flat information into hierarchical, semantically
    organized knowledge structures. It bridges the gap between raw information
    retrieval and coherent content generation.
    
    Key capabilities:
    1. Hierarchical organization of information
    2. Semantic clustering of related concepts  
    3. Integration with multiple information sources
    4. Template generation for content creation
    
    Think of this as an intelligent librarian that not only stores information
    but organizes it in ways that make it easier to find and use related concepts.
    """

    def __init__(self, topic: str):
        self.topic = topic
        self.root = KnowledgeNode(title=topic)
        self.wiki_retriever = WikipediaRetriever()
        
        # Initialize semantic embedding model for intelligent organization
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"KnowledgeBase initialized for topic: '{topic}'")

    def populate_from_wikipedia(self, max_articles: int = 3, max_sections: int = 5) -> int:
        """
        Populate knowledge base with Wikipedia content.
        
        This method demonstrates how to integrate with specialized retrievers
        while maintaining the knowledge base's organizational capabilities.
        """
        print(f"Populating knowledge base from Wikipedia...")
        
        wiki_content = self.wiki_retriever.get_wiki_content(
            self.topic, max_articles=max_articles, max_sections=max_sections
        )

        for snippet in wiki_content:
            self.root.add_snippet(snippet)

        print(f"Added {len(wiki_content)} Wikipedia snippets to knowledge base")
        return len(wiki_content)

    def populate_from_search_results(self, search_results: List[SearchResult]) -> int:
        """
        NEW METHOD: Populate knowledge base from SearchEngine results.
        
        This is the bridge method that connects your knowledge organization
        capabilities with the unified SearchEngine. It demonstrates how to
        adapt existing components to work with new interfaces.
        
        Args:
            search_results: Results from SearchEngine.search()
            
        Returns:
            Number of results added to the knowledge base
        """
        print(f"Populating knowledge base from {len(search_results)} search results...")
        
        for result in search_results:
            # Convert SearchResult to snippet format
            snippet = {
                "content": result.content,
                "source": result.source,
                "relevance_score": result.relevance_score,
                "retrieval_method": "search_engine",
                "title": "Search Result",  # SearchResult doesn't have title field
                "section": "Content"
            }
            
            self.root.add_snippet(snippet)

        print(f"Added {len(search_results)} search results to knowledge base")
        return len(search_results)

    def organize_knowledge_hierarchically(self, num_main_topics: int = 5) -> None:
        """
        Organize flat information into hierarchical structure using LLM assistance.
        
        This is where the real magic happens - transforming a flat list of
        information snippets into a meaningful knowledge hierarchy. This process
        demonstrates how to combine LLM reasoning with programmatic organization.
        
        The approach:
        1. Use LLM to identify main conceptual categories
        2. Use semantic similarity to assign content to categories
        3. Create hierarchical structure with main topics and subtopics
        """
        print("Organizing knowledge into hierarchical structure...")

        all_snippets = self.root.snippets
        if not all_snippets:
            print("No content to organize")
            return

        # Prepare content for LLM analysis
        # We limit the content length to work within LLM context windows
        snippets_text = "\n\n".join([
            f"Snippet {i+1}: {snippet['content'][:500]}..." 
            for i, snippet in enumerate(all_snippets[:10])  # Limit to 10 snippets
        ])

        # Use LLM to identify main organizational categories
        organization_prompt = f"""
        Analyze the following information about '{self.topic}' and organize it into {num_main_topics} main conceptual categories.
        
        Information to organize:
        {snippets_text}
        
        For each main category, suggest 2-3 subcategories that would help organize the information effectively.
        
        Respond with ONLY a JSON object in this format:
        {{
            "categories": [
                {{
                    "main_topic": "Category Name",
                    "subcategories": ["Subcategory 1", "Subcategory 2"]
                }}
            ]
        }}
        """

        try:
            # This would call your LLM API - for now, we'll use a structured fallback
            # In your actual implementation, replace this with your API call
            organization_response = self._call_llm_for_organization(organization_prompt)
            
            # Parse LLM response and create new knowledge structure
            new_structure = self._create_hierarchical_structure(organization_response)
            
            # Assign existing content to new structure using semantic similarity
            self._assign_content_semantically(all_snippets, new_structure)
            
            # Replace root with new organized structure
            self.root = new_structure
            
            print(f"Successfully organized knowledge into {len(self.root.children)} main categories")
            
        except Exception as e:
            print(f"LLM organization failed: {e}, using fallback structure")
            self._create_fallback_structure(all_snippets)

    def _call_llm_for_organization(self, prompt: str) -> str:
        """
        Call LLM for knowledge organization.
        
        This method would integrate with your existing LLM API calling infrastructure.
        For now, it provides a structured fallback, but you'll replace this with
        your actual API integration.
        """
        # TODO: Replace this with your actual LLM API call
        # return your_api_client.call_api(prompt)
        
        # Fallback response for demonstration
        return '''
        {
            "categories": [
                {
                    "main_topic": "Fundamental Concepts",
                    "subcategories": ["Definitions", "Core Principles"]
                },
                {
                    "main_topic": "Technical Details", 
                    "subcategories": ["Implementation", "Architecture"]
                },
                {
                    "main_topic": "Applications",
                    "subcategories": ["Current Uses", "Future Potential"]
                },
                {
                    "main_topic": "Challenges and Considerations",
                    "subcategories": ["Limitations", "Ethical Issues"]
                }
            ]
        }
        '''

    def _create_hierarchical_structure(self, llm_response: str) -> KnowledgeNode:
        """
        Create hierarchical knowledge structure from LLM analysis.
        
        This method demonstrates how to parse LLM outputs and create
        structured data representations from natural language responses.
        """
        try:
            # Extract JSON from LLM response
            import json
            
            # Handle responses that might have explanatory text around the JSON
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_text = llm_response[json_start:json_end]
                organization_data = json.loads(json_text)
            else:
                raise ValueError("No JSON found in LLM response")

            # Create new root node
            new_root = KnowledgeNode(title=self.topic)

            # Create hierarchical structure from LLM analysis
            for category_data in organization_data.get("categories", []):
                # Create main category node
                main_topic = category_data.get("main_topic", "Unknown Category")
                category_node = KnowledgeNode(title=main_topic)
                new_root.add_child(category_node)

                # Create subcategory nodes
                for subcategory in category_data.get("subcategories", []):
                    subcategory_node = KnowledgeNode(title=subcategory)
                    category_node.add_child(subcategory_node)

            return new_root

        except Exception as e:
            print(f"Failed to parse LLM organization response: {e}")
            return self._create_default_structure()

    def _create_default_structure(self) -> KnowledgeNode:
        """Create a sensible default knowledge structure."""
        new_root = KnowledgeNode(title=self.topic)
        
        default_categories = [
            ("Overview and Introduction", ["Definition", "Key Concepts"]),
            ("Technical Foundation", ["Core Components", "How It Works"]),
            ("Real-world Applications", ["Current Uses", "Case Studies"]),
            ("Future Directions", ["Emerging Trends", "Potential Impact"])
        ]
        
        for main_topic, subcategories in default_categories:
            category_node = KnowledgeNode(title=main_topic)
            new_root.add_child(category_node)
            
            for subcategory in subcategories:
                subcategory_node = KnowledgeNode(title=subcategory)
                category_node.add_child(subcategory_node)
        
        return new_root

    def _assign_content_semantically(self, snippets: List[Dict[str, Any]], structure: KnowledgeNode):
        """
        Assign content snippets to knowledge structure using semantic similarity.
        
        This method demonstrates how to use semantic embeddings to intelligently
        organize content based on meaning rather than just keywords.
        """
        print("Assigning content using semantic similarity...")
        
        # Calculate embeddings for all nodes in the structure
        all_nodes = [structure] + self._get_all_nodes(structure)
        for node in all_nodes:
            node.calculate_semantic_embedding(self.embedder)

        # Assign each snippet to the most semantically similar node
        for snippet in snippets:
            snippet_content = snippet.get('content', '')
            if not snippet_content:
                continue
                
            snippet_embedding = self.embedder.encode(snippet_content)
            
            best_node = structure
            best_similarity = 0
            
            # Find the most semantically similar node
            for node in all_nodes:
                if node.semantic_embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(snippet_embedding, node.semantic_embedding) / (
                        np.linalg.norm(snippet_embedding) * np.linalg.norm(node.semantic_embedding)
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_node = node
            
            # Assign snippet to best matching node
            best_node.add_snippet(snippet)

    def _get_all_nodes(self, root: KnowledgeNode) -> List[KnowledgeNode]:
        """Get all nodes in the knowledge structure (depth-first traversal)."""
        nodes = []
        for child in root.children:
            nodes.append(child)
            nodes.extend(self._get_all_nodes(child))
        return nodes

    def to_outline(self) -> Dict[str, Any]:
        """
        Convert knowledge base to outline structure for content generation.
        
        This method provides a clean interface for content generation workflows
        to access the organized knowledge in a format suitable for article creation.
        """
        sections = []

        # Convert each main category to an outline section
        for category_node in self.root.children:
            section = {
                "title": category_node.title,
                "key_points": [],
                "subtopics": [child.title for child in category_node.children],
                "content_summary": ""
            }

            # Extract key points from content snippets
            all_content = category_node.get_all_content()
            if all_content:
                # Extract first sentence from each paragraph as key points
                sentences = re.split(r'(?<=[.!?])\s+', all_content)
                meaningful_sentences = [
                    sentence.strip() for sentence in sentences[:3] 
                    if len(sentence.strip()) > 20
                ]
                section["key_points"] = meaningful_sentences
                section["content_summary"] = all_content[:300] + "..." if len(all_content) > 300 else all_content

            sections.append(section)

        return {
            "title": self.topic,
            "sections": sections,
            "total_content_nodes": len(self._get_all_nodes(self.root)),
            "organization_method": "semantic_hierarchical"
        }

    def get_all_content(self) -> str:
        """Get all content from the knowledge base concatenated."""
        return self.root.get_all_content()

    def get_sources(self) -> List[str]:
        """Get list of all information sources used."""
        sources = set()
        
        def collect_sources(node: KnowledgeNode):
            for snippet in node.snippets:
                if "source" in snippet:
                    sources.add(snippet["source"])
                elif "title" in snippet and "url" in snippet:
                    sources.add(f"{snippet['title']} ({snippet['url']})")
            
            for child in node.children:
                collect_sources(child)
        
        collect_sources(self.root)
        return list(sources)

    def clear(self):
        """Clear all content from knowledge base."""
        self.root = KnowledgeNode(title=self.topic)
        
    def get_content_for_section(self, section_title: str) -> str:
        """
        Get relevant content for a specific section.
        
        This method enables targeted content retrieval for section-by-section
        content generation, supporting more focused and relevant writing.
        """
        relevant_content = []
        
        def search_nodes(node: KnowledgeNode, target_title: str):
            # Check if this node matches the section title
            if (target_title.lower() in node.title.lower() or 
                node.title.lower() in target_title.lower()):
                
                # Collect content from this node and its children
                content = node.get_all_content()
                if content:
                    relevant_content.append(content)
            
            # Recursively search children
            for child in node.children:
                search_nodes(child, target_title)
        
        search_nodes(self.root, section_title)
        
        return "\n\n".join(relevant_content) if relevant_content else ""


# Example integration with SearchEngine
def create_knowledge_base_from_search(topic: str, search_engine) -> KnowledgeBase:
    """
    Convenience function showing how to integrate KnowledgeBase with SearchEngine.
    
    This function demonstrates the clean integration pattern between your
    information retrieval system (SearchEngine) and knowledge organization
    system (KnowledgeBase).
    
    Args:
        topic: The topic to build knowledge about
        search_engine: An instance of your SearchEngine
        
    Returns:
        Organized KnowledgeBase ready for content generation
    """
    # Step 1: Gather information using SearchEngine
    search_results = search_engine.search(topic, num_results=15)
    
    # Step 2: Create and populate KnowledgeBase
    kb = KnowledgeBase(topic)
    
    # Add search results to knowledge base
    kb.populate_from_search_results(search_results)
    
    # Also add Wikipedia content for comprehensive coverage
    kb.populate_from_wikipedia(max_articles=3, max_sections=5)
    
    # Step 3: Organize the collected information intelligently
    kb.organize_knowledge_hierarchically(num_main_topics=5)
    
    return kb