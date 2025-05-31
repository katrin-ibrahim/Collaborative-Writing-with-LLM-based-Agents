from typing import List, Dict
import logging
from ..base_agent import BaseAgent
from ...utils.data_models import Article, Outline, SearchResult
from ...retrieval.search_engine import SearchEngine
from ...retrieval.passage_ranker import PassageRanker
from .outline_generator import OutlineGenerator

logger = logging.getLogger(__name__)

class WriterAgent(BaseAgent):
    """Main writer agent responsible for content creation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.search_engine = SearchEngine()
        self.passage_ranker = PassageRanker(top_k=config.get('retrieval.top_k', 5))
        self.outline_generator = OutlineGenerator(config)
    
    def process(self, topic: str) -> Article:
        """
        Main processing pipeline for content generation.
        
        This implements the Writer's workflow:
        1. Issue retrieval queries
        2. Collate top-k passages
        3. Draft hierarchical outline
        4. Expand each heading into content
        """
        logger.info(f"Starting content generation for topic: {topic}")
        
        # Step 1: Retrieve relevant information
        search_results = self._retrieve_information(topic)
        
        # Step 2: Create working context from top passages
        context = self.passage_ranker.create_context(search_results)
        
        # Step 3: Generate outline
        outline = self.outline_generator.generate_outline(topic, context)
        
        # Step 4: Expand outline into full article
        article = self._expand_outline_to_article(outline, context)
        
        logger.info(f"Completed article generation: {len(article.content)} characters")
        return article
    
    def _retrieve_information(self, topic: str) -> List[SearchResult]:
        """Retrieve and rank relevant information for the topic."""
        
        # Generate multiple search queries for comprehensive coverage
        search_queries = self._generate_search_queries(topic)
        
        all_results = []
        for query in search_queries:
            results = self.search_engine.search(query, num_results=10)
            all_results.extend(results)
        
        # Rank and select top passages
        top_results = self.passage_ranker.rank_passages(all_results, topic)
        
        return top_results
    
    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate multiple search queries for comprehensive information retrieval."""
        
        # Create diverse queries to capture different aspects
        queries = [
            topic,  # Direct topic search
            f"{topic} definition explanation",  # Definitional
            f"{topic} current developments news",  # Current events
            f"{topic} background history",  # Historical context
            f"{topic} examples applications"  # Practical examples
        ]
        
        return queries
    
    def _expand_outline_to_article(self, outline: Outline, context: str) -> Article:
        """Expand the outline into a full article."""
        
        sections = {}
        full_content_parts = [f"# {outline.title}\n"]
        
        for heading in outline.headings:
            # Create section-specific context
            section_context = self._create_section_context(heading, context)
            
            # Generate content for this section
            section_content = self._generate_section_content(
                heading, 
                outline.subheadings.get(heading, []),
                section_context
            )
            
            sections[heading] = section_content
            full_content_parts.append(f"\n## {heading}\n{section_content}")
        
        full_content = "\n".join(full_content_parts)
        
        return Article(
            title=outline.title,
            content=full_content,
            outline=outline,
            sections=sections,
            metadata={"word_count": len(full_content.split())}
        )
    
    def _create_section_context(self, heading: str, full_context: str) -> str:
        """Extract relevant context for a specific section."""
        # Simple approach: return full context
        # In advanced implementation, would use similarity matching
        return full_context
    
    def _generate_section_content(self, heading: str, subheadings: List[str], context: str) -> str:
        """Generate content for a specific section."""
        
        subheading_text = ", ".join(subheadings) if subheadings else ""
        
        section_prompt = f"""
        Write a comprehensive section about: {heading}
        
        {"Cover these subtopics: " + subheading_text if subheading_text else ""}
        
        Use this relevant information:
        {context}
        
        Write 2-3 well-structured paragraphs that:
        1. Introduce the topic clearly
        2. Provide detailed explanations with examples
        3. Connect to the broader context
        
        Be informative, accurate, and engaging.
        """
        
        content = self.call_api(section_prompt)
        
        # Clean and format the response
        content = content.strip()
        if not content:
            content = f"This section covers {heading}, providing important insights and detailed information about this topic."
        
        return content
