from typing import Dict, List
from knowledge.knowledge_base import KnowledgeBase
from utils.data_models import SearchResult


class KnowledgeToolkit:
    """
    Knowledge organization tools that can be used by any agent.
    
    Different agents might organize knowledge differently:
    - Writer: Organize for content creation flow
    - Reviewer: Organize for fact-checking and verification
    """
    
    def create_organizer(self, topic: str) -> KnowledgeBase:
        """Create a new knowledge organizer for any purpose."""
        return KnowledgeBase(topic)
    
    def organize_for_writing(self, organizer: KnowledgeBase, 
                            categories: List[Dict[str, str]], 
                            search_results: List[SearchResult]) -> None:
        """Organize knowledge optimized for content creation."""
        self._organize_by_categories(organizer, categories, search_results)
    
    def organize_for_verification(self, organizer: KnowledgeBase,
                                 claims_to_verify: List[str],
                                 search_results: List[SearchResult]) -> None:
        """
        Organize knowledge optimized for fact-checking.
        Future: This will help reviewer agents verify specific claims.
        """
        verification_categories = [
            {"name": "Factual Claims", "description": "Verifiable facts and statistics"},
            {"name": "Source Evidence", "description": "Supporting evidence from reliable sources"},
            {"name": "Contradictions", "description": "Conflicting information that needs resolution"},
            {"name": "Uncertain Claims", "description": "Claims that need additional verification"}
        ]
        
        self._organize_by_categories(organizer, verification_categories, search_results)

    def _organize_by_categories(self, organizer: KnowledgeBase, 
                               categories: List[Dict[str, str]], 
                               search_results: List[SearchResult]) -> None:
        """Internal method for category-based organization."""
        category_nodes = {}
        for category in categories:
            node = organizer.create_category(category["name"], category.get("description", ""))
            category_nodes[category["name"]] = node
        
        for category_name, node in category_nodes.items():
            similar_results = organizer.find_most_similar_results(
                category_name + " " + node.content, 
                search_results, 
                top_k=max(1, len(search_results) // len(categories))
            )
            organizer.assign_results_to_category(similar_results, node)
    
    def find_relevant_content(self, organizer: KnowledgeBase, 
                             target_description: str, 
                             max_results: int = 5) -> List[SearchResult]:
        """Find content most relevant to target description."""
        all_results = organizer.get_all_search_results()
        return organizer.find_most_similar_results(target_description, all_results, max_results)
    
    def extract_claims_from_content(self, content: str) -> List[str]:
        """
        Extract verifiable claims from content.
        Future: Critical for reviewer agents to identify what needs fact-checking.
        """
        import re
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Simple heuristic for finding factual claims
        # Future: Replace with more sophisticated claim detection
        claims = []
        for sentence in sentences:
            if (any(word in sentence.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have']) and
                len(sentence.split()) > 5 and
                not sentence.strip().startswith(('However', 'Moreover', 'Furthermore', 'In conclusion'))):
                claims.append(sentence.strip())
        
        return claims[:10]  # Limit for practical use