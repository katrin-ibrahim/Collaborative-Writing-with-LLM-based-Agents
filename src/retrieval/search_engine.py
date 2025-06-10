# src/retrieval/search_engine.py - Clean, unified search engine
import requests
from typing import List, Dict, Any, Optional
import logging
from utils.data_models import SearchResult
from knowledge.knowledge_base import WikipediaRetriever
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Unified search engine that handles all knowledge retrieval needs.
    
    This class is responsible for gathering information for content generation,
    NOT for evaluation. FreshWiki is handled separately in the evaluation module.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {'retrieval': {'top_k': 5}}
        
        self.config = config
        self.session = requests.Session()
        self.wiki_retriever = WikipediaRetriever()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        logger.info("SearchEngine initialized with Wikipedia retrieval capability")
    
    def search_all_sources(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        Perform comprehensive search for content generation.
        
        This method gathers information to help generate content
        It combines multiple knowledge sources to provide comprehensive information.
        """
        
        all_results = []
        
        # Strategy 1: Use Wikipedia as primary knowledge source
        wiki_results = self._search_wikipedia(query, num_results // 2)
        all_results.extend(wiki_results)

        
        # Strategy 2: Mock web search (replace with real API when available)
        # TODO: Integrate actual web search APIs like DuckDuckGo, Bing, Google
        if len(all_results) < num_results:
            remaining = num_results - len(all_results)
            web_results = self._mock_web_search(query, remaining)
            all_results.extend(web_results)
        
        # Remove duplicates and rank by relevance
        unique_results = self._deduplicate_and_rank(all_results)
        
        logger.info(f"Retrieved {len(unique_results)} search results for query: {query}")
        return unique_results[:num_results]
    
    def _search_wikipedia(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search Wikipedia using existing retriever.
        
        This provides factual, encyclopedic knowledge for content generation.
        """
        try:
            entity_variations = self._generate_entity_focused_queries(query)
            all_snippets = []
            for variation in entity_variations:
                snippets = self.wiki_retriever.get_wiki_content(
                    variation, max_articles=max(1, max_results // len(entity_variations)), max_sections=2
                )
                all_snippets.extend(snippets)
       
            
            search_results = []
            for snippet in all_snippets:
                # Convert snippet format to SearchResult
                content = snippet.get('content', '')
                title = snippet.get('title', '')
                section = snippet.get('section', '')
                url = snippet.get('url', '')
                
                # Create descriptive source attribution
                source = f"Wikipedia: {title}"
                if section and section != 'Summary':
                    source += f" - {section}"
                
                # Calculate relevance using semantic similarity
                relevance = self._calculate_relevance(query, content)
                
                search_results.append(SearchResult(
                    content=content,
                    source=source,
                    relevance_score=relevance
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Wikipedia search failed for query '{query}': {e}")
            return []
        
    # TODO: Implement more sophisticated query generation strategies
    def _generate_entity_focused_queries(self, query: str) -> List[str]:
        base_queries = [query]
        
        topic_expansions = [
            f"{query} history background",
            f"{query} key people figures",
            f"{query} important dates timeline",
            f"{query} locations geography",
            f"{query} technical specifications",
            f"{query} current developments recent"
        ]
        
        base_queries.extend(topic_expansions[:3])
        return base_queries
    
    def _mock_web_search(self, query: str, num_results: int) -> List[SearchResult]:
        """
        Mock web search implementation.
        
        TODO replace this with actual search API calls to:
        - DuckDuckGo API
        - Bing Search API  
        - Google Custom Search API
        """
        try:
            mock_results = []
            for i in range(min(num_results, 3)):  # Limit mock results
                mock_results.append(SearchResult(
                    content=f"Web search result {i+1} for '{query}'. "
                            f"This would contain current, relevant information from web sources "
                            f"that complements the encyclopedic knowledge from Wikipedia. "
                            f"Real implementation would call actual search APIs.",
                    source=f"Web Search - Source {i+1}",
                    relevance_score=0.7 - (i * 0.1)  # Decreasing relevance
                ))
            
            return mock_results
            
        except Exception as e:
            logger.error(f"Mock web search failed: {e}")
            return []
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """
        Calculate semantic relevance between query and content.
        
        This helps rank search results by how well they match the information need.
        """
        try:
            query_embedding = self.embedder.encode(query)
            content_embedding = self.embedder.encode(content)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, content_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
            )
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.warning(f"Relevance calculation failed: {e}")
            return 0.5  # Default relevance score
    
    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate content and rank by relevance.
        
        This ensures we provide diverse, high-quality information sources.
        """
        if len(results) <= 1:
            return results
        
        # Simple deduplication based on content similarity
        unique_results = []
        seen_content_hashes = set()
        
        for result in results:
            # Create hash from first 200 characters for deduplication
            content_hash = hash(result.content[:200].lower().strip())
            
            if content_hash not in seen_content_hashes:
                seen_content_hashes.add(content_hash)
                unique_results.append(result)
        
        # Sort by relevance score (highest first)
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return unique_results


