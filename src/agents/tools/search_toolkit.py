# src/agents/tools/agent_toolkit.py
from typing import List, Dict, Any
from utils.data_models import SearchResult
from retrieval.search_engine import SearchEngine

class SearchToolkit:
    """
    Search tools that can be used by any agent (writer, reviewer, etc.).
    
    These tools are agent-agnostic and focus purely on information retrieval.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.search_engine = SearchEngine(config)
    
    def search_wikipedia(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search Wikipedia and return results."""
        return self.search_engine.search_wikipedia(query, max_results)
    
    def search_web(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search web and return results."""
        return self.search_engine.search_web(query, max_results)
    
    def search_all_sources(self, query: str, wiki_results: int = 3, web_results: int = 3) -> List[SearchResult]:
        """Search all available sources."""
        return self.search_engine.search_all_sources(query, wiki_results, web_results)
    
    def verify_claim(self, claim: str, context_results: List[SearchResult] = None) -> Dict[str, Any]:
        """
        Search for information to verify a specific claim.
        Useful for both content generation and fact-checking.
        """
        verification_query = f"verify {claim}"
        search_results = self.search_all_sources(verification_query, wiki_results=2, web_results=3)
        
        return {
            "claim": claim,
            "verification_results": search_results,
            "evidence_count": len(search_results),
            "sources": [result.source for result in search_results]
        }