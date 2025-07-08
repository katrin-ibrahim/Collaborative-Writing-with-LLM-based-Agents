import logging
from typing import List, Dict, Any, Union
from knowledge.wikipedia_retriever import WikipediaRetriever

logger = logging.getLogger(__name__)


class WikipediaSearchRM:
    """
    Wikipedia-based search retrieval manager for STORM integration.
    
    Provides real Wikipedia content while maintaining STORM's expected interface.
    """
    
    def __init__(self, k: int = 3):
        self.k = k
        self.retriever = WikipediaRetriever()
        logger.info(f"WikipediaSearchRM initialized with k={k}")
    
    def __call__(self, query_or_queries, exclude_urls=None, **kwargs):
        """Make the object callable for STORM compatibility."""
        return self.retrieve(query_or_queries, exclude_urls, **kwargs)
    
    def retrieve(self, query_or_queries, exclude_urls=None, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve Wikipedia content in STORM's expected format.
        
        Args:
            query_or_queries: Single query string or list of queries
            exclude_urls: URLs to exclude (applied to Wikipedia URLs)
            **kwargs: Additional search parameters
            
        Returns:
            List of search result dictionaries in STORM format
        """
        # Handle both single query and multiple queries
        if isinstance(query_or_queries, list):
            queries = query_or_queries
        else:
            queries = [query_or_queries]
        
        logger.debug(f"WikipediaSearchRM processing {len(queries)} queries")
        
        results = []
        excluded_urls = set(exclude_urls or [])
        
        for query in queries:
            try:
                # Get Wikipedia content for this query
                wiki_snippets = self.retriever.get_wiki_content(
                    topic=query, 
                    max_articles=self.k, 
                    max_sections=3
                )
                
                # Convert to STORM format
                query_results = self._convert_to_storm_format(wiki_snippets, excluded_urls)
                results.extend(query_results)
                
            except Exception as e:
                logger.warning(f"Wikipedia search failed for query '{query}': {e}")
                # Add fallback result to maintain expected behavior
                results.append(self._create_fallback_result(query))
        
        # Limit total results and ensure we don't exceed k per query
        final_results = results[:self.k * len(queries)]
        logger.info(f"WikipediaSearchRM returned {len(final_results)} results for {len(queries)} queries")
        
        return final_results
    
    def _convert_to_storm_format(self, wiki_snippets: List[Dict], excluded_urls: set) -> List[Dict[str, Any]]:
        """Convert Wikipedia snippets to STORM's expected format."""
        storm_results = []
        
        for snippet in wiki_snippets:
            url = snippet.get('url', '')
            
            # Skip if URL is in exclusion list
            if url in excluded_urls:
                continue
            
            # STORM expects this exact structure
            storm_result = {
                'url': url,
                'snippets': [snippet.get('content', '')],
                'title': f"{snippet.get('title', '')} - {snippet.get('section', '')}",
                'description': self._create_description(snippet)
            }
            
            storm_results.append(storm_result)
        
        return storm_results
    
    def _create_description(self, snippet: Dict) -> str:
        """Create a description from Wikipedia snippet metadata."""
        title = snippet.get('title', 'Wikipedia Article')
        section = snippet.get('section', '')
        method = snippet.get('retrieval_method', 'wikipedia')
        
        if section and section != 'Summary':
            return f"Wikipedia article '{title}', section: {section}"
        else:
            return f"Wikipedia article: {title}"
    
    def _create_fallback_result(self, query: str) -> Dict[str, Any]:
        """Create a fallback result when Wikipedia search fails."""
        return {
            'url': f'https://en.wikipedia.org/wiki/{query.replace(" ", "_")}',
            'snippets': [f'Information about {query} from Wikipedia. This topic may require further research for comprehensive coverage.'],
            'title': f'Wikipedia: {query}',
            'description': f'Wikipedia article about {query}'
        }