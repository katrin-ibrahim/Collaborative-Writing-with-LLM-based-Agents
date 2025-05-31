from typing import List, Dict, Union
import logging
from agents.base_agent import BaseAgent
from utils.data_models import Article, Outline, SearchResult
from retrieval.search_engine import SearchEngine
from retrieval.passage_ranker import PassageRanker
from knowledge.knowledge_base import KnowledgeBase, create_knowledge_base_from_search
from .outline_generator import OutlineGenerator

logger = logging.getLogger(__name__)

class WriterAgent(BaseAgent):
    """
    Comprehensive writing agent with configurable capabilities.
    
    This agent embodies the full range of content generation capabilities,
    from simple outline-based writing to sophisticated knowledge-enhanced
    generation. The agent adapts its approach based on configuration,
    making it suitable for different experimental conditions while
    maintaining consistent core writing intelligence.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Core writing capabilities - always available
        self.outline_generator = OutlineGenerator(config)
        
        # Knowledge retrieval capabilities - configurable
        self.use_external_knowledge = config.get('writer.use_external_knowledge', True)
        self.use_knowledge_base = config.get('writer.use_knowledge_base', True)
        self.knowledge_organization_depth = config.get('writer.knowledge_depth', 'semantic_hierarchical')
        
        # Initialize retrieval components only if needed
        if self.use_external_knowledge:
            self.search_engine = SearchEngine()
            self.passage_ranker = PassageRanker(top_k=config.get('retrieval.top_k', 5))
        else:
            self.search_engine = None
            self.passage_ranker = None
        
        logger.info(f"WriterAgent initialized with external_knowledge={self.use_external_knowledge}, knowledge_base={self.use_knowledge_base}")
    
    def process(self, topic: str) -> Article:
        """
        Main processing pipeline that adapts based on agent configuration.
        
        This method demonstrates how an agent can maintain consistent behavior
        while adapting its approach based on available tools and constraints.
        The core writing process remains the same, but the knowledge gathering
        and organization steps adapt to the agent's configuration.
        """
        logger.info(f"WriterAgent processing topic: {topic}")
        
        # Step 1: Gather knowledge using configured approach
        knowledge_context = self._gather_knowledge(topic)
        
        # Step 2: Generate outline informed by available knowledge
        outline = self._generate_informed_outline(topic, knowledge_context)
        
        # Step 3: Expand outline using available knowledge
        article = self._expand_outline_with_context(outline, knowledge_context, topic)
        
        logger.info(f"WriterAgent completed processing: {len(article.content)} characters")
        return article
    
    def _gather_knowledge(self, topic: str) -> Union[KnowledgeBase, List[SearchResult], None]:
        """
        Gather knowledge using the approach configured for this agent.
        
        This method embodies the agent's adaptability - it can work with
        sophisticated knowledge organization, simple retrieval, or purely
        internal knowledge, depending on configuration and available tools.
        """
        if not self.use_external_knowledge:
            # Agent working from internal knowledge only
            logger.info("Agent operating in internal-knowledge-only mode")
            return None
        
        if self.use_knowledge_base and self.knowledge_organization_depth == 'semantic_hierarchical':
            # Agent using full knowledge organization capabilities
            logger.info("Agent using semantic hierarchical knowledge organization")
            kb = create_knowledge_base_from_search(topic, self.search_engine)
            return kb
        
        elif self.use_external_knowledge:
            # Agent using basic retrieval without organization
            logger.info("Agent using basic knowledge retrieval")
            search_results = self._perform_basic_retrieval(topic)
            top_results = self.passage_ranker.rank_passages(search_results, topic)
            return top_results
        
        return None
    
    def _perform_basic_retrieval(self, topic: str) -> List[SearchResult]:
        """Perform basic multi-query retrieval without organization."""
        search_queries = [
            topic,
            f"{topic} definition explanation",
            f"{topic} current developments",
            f"{topic} background context",
            f"{topic} examples applications"
        ]
        
        all_results = []
        for query in search_queries:
            results = self.search_engine.search(query, num_results=8)
            all_results.extend(results)
        
        return all_results
    
    def _generate_informed_outline(self, topic: str, knowledge_context) -> Outline:
        """
        Generate outline informed by whatever knowledge is available.
        
        This method shows how the agent maintains consistent outline generation
        capabilities while adapting to different types of knowledge input.
        """
        if isinstance(knowledge_context, KnowledgeBase):
            # Use organized knowledge to inform outline structure
            outline_dict = knowledge_context.to_outline()
            return Outline(
                title=outline_dict['title'],
                headings=[section['title'] for section in outline_dict['sections']],
                subheadings={section['title']: section.get('key_points', []) 
                           for section in outline_dict['sections']}
            )
        
        elif isinstance(knowledge_context, list) and knowledge_context:
            # Use search results to inform outline
            context_text = self.passage_ranker.create_context(knowledge_context)
            return self.outline_generator.generate_outline(topic, context_text)
        
        else:
            # Generate outline from internal knowledge only
            return self.outline_generator.generate_outline(topic, "")
    
    def _expand_outline_with_context(self, outline: Outline, knowledge_context, topic: str) -> Article:
        """
        Expand outline using whatever knowledge context is available.
        
        This method demonstrates the agent's ability to maintain consistent
        content generation quality while working with different types of
        knowledge input, from sophisticated knowledge bases to simple context.
        """
        sections_content = {}
        content_parts = [f"# {outline.title}\n"]
        
        for heading in outline.headings:
            section_context = self._get_section_specific_context(heading, knowledge_context)
            section_subheadings = outline.subheadings.get(heading, [])
            
            section_content = self._generate_section_content_with_context(
                heading, section_subheadings, section_context, outline.title
            )
            
            sections_content[heading] = section_content
            content_parts.append(f"\n## {heading}\n\n{section_content}")
        
        full_content = "\n".join(content_parts)
        
        # Create metadata that reflects the agent's actual capabilities used
        metadata = {
            "method": "writer_agent",
            "word_count": len(full_content.split()),
            "external_knowledge_used": self.use_external_knowledge,
            "knowledge_organization": self.knowledge_organization_depth if self.use_knowledge_base else "none"
        }
        
        if isinstance(knowledge_context, KnowledgeBase):
            metadata["sources"] = knowledge_context.get_sources()
        elif isinstance(knowledge_context, list):
            metadata["num_passages"] = len(knowledge_context)
        
        return Article(
            title=outline.title,
            content=full_content,
            outline=outline,
            sections=sections_content,
            metadata=metadata
        )
    
    def _get_section_specific_context(self, heading: str, knowledge_context) -> str:
        """Extract relevant context for a specific section based on available knowledge."""
        if isinstance(knowledge_context, KnowledgeBase):
            return knowledge_context.get_content_for_section(heading)
        elif isinstance(knowledge_context, list) and knowledge_context:
            # Simple approach for basic retrieval - return general context
            return self.passage_ranker.create_context(knowledge_context[:3])
        else:
            return ""
    
    def _generate_section_content_with_context(self, heading: str, subheadings: List[str], 
                                             context: str, article_title: str) -> str:
        """Generate section content using available context."""
        subheading_text = ", ".join(subheadings) if subheadings else ""
        
        if context:
            # Agent has external knowledge to work with
            section_prompt = f"""
            Write a comprehensive section titled "{heading}" for an article about "{article_title}".
            
            {"Cover these subtopics: " + subheading_text if subheading_text else ""}
            
            Use this relevant information as foundation:
            {context}
            
            Requirements:
            1. Write 2-3 well-structured paragraphs (300-400 words)
            2. Ground your writing in the provided information
            3. Maintain an informative, engaging tone
            4. Include specific details and examples from the context
            5. Ensure logical flow and clear organization
            """
        else:
            # Agent working from internal knowledge only
            section_prompt = f"""
            Write a comprehensive section titled "{heading}" for an article about "{article_title}".
            
            {"Cover these subtopics: " + subheading_text if subheading_text else ""}
            
            Requirements:
            1. Write 2-3 well-structured paragraphs (300-400 words)
            2. Draw from your knowledge of {article_title}
            3. Provide clear explanations and relevant examples
            4. Maintain an informative, engaging tone
            5. Ensure logical flow and organization
            
            Focus on creating valuable content that demonstrates understanding of {heading}
            in the context of {article_title}.
            """
        
        content = self.call_api(section_prompt)
        return content.strip() if content.strip() else f"Content for {heading} in the context of {article_title}."