from typing import Dict, List
import logging
from .base_workflow import BaseWorkflow
from ..agents.writer.writer_agent import WriterAgent
from ..utils.data_models import Article, SearchResult
from ..retrieval.search_engine import SearchEngine
from ..retrieval.passage_ranker import PassageRanker

logger = logging.getLogger(__name__)

class RAGWriterWorkflow(BaseWorkflow):
    """
    Baseline 3: RAG + Writer
    Combines retrieval-augmented generation with structured writing approach.
    This integrates your knowledge base concepts with the baseline structure.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.search_engine = SearchEngine()
        self.passage_ranker = PassageRanker(top_k=config.get('retrieval.top_k', 5))
        self.writer_agent = WriterAgent(config)
        
    def generate_content(self, topic: str) -> Article:
        """Generate article using RAG + structured writing approach."""
        
        logger.info(f"Generating content via RAG+Writer workflow for: {topic}")
        
        # Step 1: Multi-query retrieval for comprehensive coverage
        all_search_results = self._perform_comprehensive_retrieval(topic)
        
        # Step 2: Rank and select best passages
        top_passages = self.passage_ranker.rank_passages(all_search_results, topic)
        context = self.passage_ranker.create_context(top_passages)
        
        # Step 3: Generate structured outline with retrieved context
        outline = self._generate_contextual_outline(topic, context)
        
        # Step 4: Expand outline to full article using RAG context
        article = self._expand_outline_with_rag(outline, context, top_passages)
        
        logger.info(f"RAG+Writer workflow completed: {len(article.content)} characters")
        return article
    
    def _perform_comprehensive_retrieval(self, topic: str) -> List[SearchResult]:
        """Perform multi-faceted retrieval similar to your research() method."""
        
        # Generate diverse search queries for comprehensive coverage
        search_queries = [
            topic,  # Direct topic search
            f"{topic} definition overview",  # Definitional
            f"{topic} history background",   # Historical context
            f"{topic} current developments", # Recent updates
            f"{topic} examples applications", # Practical examples
            f"{topic} importance significance" # Why it matters
        ]
        
        all_results = []
        for query in search_queries:
            try:
                results = self.search_engine.search(query, num_results=8)
                all_results.extend(results)
                logger.info(f"Retrieved {len(results)} results for query: {query}")
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
        
        return all_results
    
    def _generate_contextual_outline(self, topic: str, context: str) -> Dict:
        """Generate outline informed by retrieved context."""
        
        outline_prompt = f"""
        Create a comprehensive outline for an article about: {topic}
        
        Use this retrieved information to inform your structure:
        {context[:2000]}...  # Truncate for prompt length
        
        Generate an outline with:
        1. A compelling title
        2. 4-6 main sections that logically organize the topic
        3. 2-3 key points for each section
        
        Format as:
        Title: [Article Title]
        
        Section 1: [Section Name]
        - Key point 1
        - Key point 2
        
        Section 2: [Section Name]
        - Key point 1
        - Key point 2
        
        Continue for all sections...
        """
        
        response = self.call_api(outline_prompt)
        return self._parse_outline_response(response, topic)
    
    def _parse_outline_response(self, response: str, topic: str) -> Dict:
        """Parse outline response into structured format."""
        lines = response.strip().split('\n')
        
        title = topic  # Default fallback
        sections = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                title = line.replace('Title:', '').strip()
            elif line.startswith('Section') and ':' in line:
                if current_section:
                    sections.append(current_section)
                section_name = line.split(':', 1)[1].strip()
                current_section = {
                    'title': section_name,
                    'key_points': []
                }
            elif line.startswith('-') and current_section:
                key_point = line.replace('-', '').strip()
                current_section['key_points'].append(key_point)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        # Fallback structure if parsing fails
        if not sections:
            sections = [
                {'title': 'Introduction', 'key_points': ['Define and introduce the topic', 'Explain its significance']},
                {'title': 'Background', 'key_points': ['Historical context', 'Key developments']},
                {'title': 'Main Concepts', 'key_points': ['Core principles', 'Important details']},
                {'title': 'Applications', 'key_points': ['Real-world examples', 'Current usage']},
                {'title': 'Future Outlook', 'key_points': ['Trends and developments', 'Potential impact']},
                {'title': 'Conclusion', 'key_points': ['Summary of key points', 'Final thoughts']}
            ]
        
        return {
            'title': title,
            'sections': sections
        }
    
    def _expand_outline_with_rag(self, outline: Dict, full_context: str, passages: List[SearchResult]) -> Article:
        """Expand outline to full article using RAG approach."""
        
        sections_content = {}
        content_parts = [f"# {outline['title']}\n"]
        
        for section in outline['sections']:
            # Find most relevant passages for this section
            section_context = self._get_section_context(section, passages)
            
            # Generate section content
            section_content = self._generate_section_with_rag(
                section, section_context, outline['title']
            )
            
            sections_content[section['title']] = section_content
            content_parts.append(f"\n## {section['title']}\n\n{section_content}")
        
        full_content = "\n".join(content_parts)
        
        return Article(
            title=outline['title'],
            content=full_content,
            sections=sections_content,
            metadata={
                "method": "rag_writer",
                "word_count": len(full_content.split()),
                "evidence_based": True,
                "num_sources": len(passages)
            }
        )
    
    def _get_section_context(self, section: Dict, passages: List[SearchResult]) -> str:
        """Extract most relevant context for a specific section."""
        
        # Simple approach: use all passages
        # Advanced implementation would use semantic similarity matching
        context_parts = []
        for passage in passages[:3]:  # Limit to top 3 for context length
            context_parts.append(f"Source: {passage.content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_section_with_rag(self, section: Dict, context: str, article_title: str) -> str:
        """Generate section content using retrieved context."""
        
        key_points_text = "\n".join(f"- {point}" for point in section['key_points'])
        
        section_prompt = f"""
        Write a comprehensive section titled "{section['title']}" for an article about "{article_title}".
        
        Cover these key points:
        {key_points_text}
        
        Use this retrieved information as evidence and context:
        {context}
        
        Requirements:
        1. Write 2-3 well-structured paragraphs (300-400 words)
        2. Ground your writing in the provided evidence
        3. Maintain an informative, engaging tone
        4. Include specific details and examples from the context
        5. Ensure smooth flow and logical organization
        
        Focus on creating content that is both informative and well-supported by evidence.
        """
        
        content = self.call_api(section_prompt)
        
        # Ensure we have content even if generation fails
        if not content.strip():
            content = f"This section covers {section['title']}, providing detailed analysis and insights based on current research and evidence about this important aspect of {article_title}."
        
        return content.strip()