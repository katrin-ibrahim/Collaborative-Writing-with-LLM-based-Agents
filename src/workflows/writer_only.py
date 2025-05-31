# src/workflows/writer_only.py - Corrected Writer-Only baseline
from typing import Dict, List
import logging
from .base_workflow import BaseWorkflow
from ..agents.writer.outline_generator import OutlineGenerator
from ..utils.data_models import Article, Outline

logger = logging.getLogger(__name__)

class WriterOnlyWorkflow(BaseWorkflow):
    """
    Baseline 2: Writer-Only workflow
    
    This approach generates content using only the language model's internal knowledge,
    without any external information retrieval. The process is:
    
    1. Generate a structured outline for the topic
    2. Expand each section of the outline into full content
    3. Combine sections into a complete article
    
    This baseline tests the LLM's ability to produce coherent, structured content
    based solely on its training knowledge, without external facts or evidence.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.outline_generator = OutlineGenerator(config)
        
        # Writer-only specific configuration
        self.target_sections = config.get('writer_only', {}).get('target_sections', 6)
        self.words_per_section = config.get('writer_only', {}).get('words_per_section', 250)
    
    def generate_content(self, topic: str) -> Article:
        """
        Generate article using outline-then-draft approach without external retrieval.
        
        This method demonstrates how well an LLM can structure and develop ideas
        using only its internal knowledge representation.
        """
        
        logger.info(f"Generating content via writer-only workflow for: {topic}")
        
        # Step 1: Generate comprehensive outline from LLM knowledge only
        outline = self._generate_comprehensive_outline(topic)
        
        # Step 2: Expand outline into full article sections
        article = self._expand_outline_to_article(outline, topic)
        
        logger.info(f"Writer-only workflow completed: {len(article.content)} characters")
        return article
    
    def _generate_comprehensive_outline(self, topic: str) -> Outline:
        """
        Generate a detailed outline using only the LLM's knowledge.
        
        This tests the model's ability to structure information logically
        and comprehensively without external guidance.
        """
        
        outline_prompt = f"""
        Create a comprehensive, well-structured outline for an informative article about: {topic}
        
        Requirements:
        1. Generate {self.target_sections} main sections that logically cover the topic
        2. Each section should have 2-3 specific subtopics
        3. Ensure logical flow from introduction to conclusion
        4. Cover both fundamental concepts and practical applications
        5. Make the outline comprehensive enough for a 1500-2000 word article
        
        Format your response as:
        Title: [Compelling Article Title]
        
        1. [Section 1 Name]
           - [Subtopic 1.1]
           - [Subtopic 1.2]
           - [Subtopic 1.3]
        
        2. [Section 2 Name]
           - [Subtopic 2.1]
           - [Subtopic 2.2]
        
        Continue for all {self.target_sections} sections...
        
        Focus on creating an outline that demonstrates deep understanding of {topic}
        and provides a logical structure for comprehensive coverage.
        """
        
        response = self.call_api(outline_prompt)
        outline = self._parse_outline_response(response, topic)
        
        logger.info(f"Generated outline with {len(outline.headings)} main sections")
        return outline
    
    def _parse_outline_response(self, response: str, topic: str) -> Outline:
        """
        Parse the LLM's outline response into structured format.
        
        This parsing demonstrates how to extract structured information
        from natural language LLM outputs.
        """
        
        lines = response.strip().split('\n')
        
        title = topic  # Default fallback
        headings = []
        subheadings = {}
        current_heading = None
        
        for line in lines:
            line = line.strip()
            
            # Extract title
            if line.lower().startswith('title:'):
                title = line.split(':', 1)[1].strip()
                continue
            
            # Check for main headings (numbered)
            if line and line[0].isdigit() and '.' in line:
                # Extract heading text after number
                heading_text = line.split('.', 1)[1].strip()
                headings.append(heading_text)
                current_heading = heading_text
                subheadings[heading_text] = []
                continue
            
            # Check for subheadings (bulleted)
            if line.startswith('-') and current_heading:
                subheading = line.replace('-', '').strip()
                if subheading:  # Only add non-empty subheadings
                    subheadings[current_heading].append(subheading)
                continue
        
        # Provide fallback structure if parsing fails
        if not headings:
            logger.warning("Outline parsing failed, using fallback structure")
            headings = [
                "Introduction and Overview",
                "Historical Background", 
                "Core Concepts and Principles",
                "Current Applications and Examples",
                "Challenges and Limitations",
                "Future Directions and Conclusion"
            ]
            subheadings = {
                heading: [
                    f"Key aspects of {heading.lower()}",
                    f"Important considerations in {heading.lower()}"
                ] for heading in headings
            }
        
        return Outline(
            title=title,
            headings=headings,
            subheadings=subheadings
        )
    
    def _expand_outline_to_article(self, outline: Outline, topic: str) -> Article:
        """
        Expand the outline into a complete article using LLM knowledge only.
        
        This process tests the model's ability to develop ideas comprehensively
        and maintain coherence across a longer piece of writing.
        """
        
        sections_content = {}
        content_parts = [f"# {outline.title}\n"]
        
        # Generate introduction
        intro_content = self._generate_introduction(topic, outline.headings)
        content_parts.append(f"{intro_content}\n")
        
        # Expand each main section
        for heading in outline.headings:
            section_subtopics = outline.subheadings.get(heading, [])
            section_content = self._generate_section_content(
                heading, section_subtopics, topic, outline.title
            )
            
            sections_content[heading] = section_content
            content_parts.append(f"\n## {heading}\n\n{section_content}")
        
        # Generate conclusion
        conclusion_content = self._generate_conclusion(topic, outline.headings)
        content_parts.append(f"\n## Conclusion\n\n{conclusion_content}")
        
        full_content = "\n".join(content_parts)
        
        return Article(
            title=outline.title,
            content=full_content,
            outline=outline,
            sections=sections_content,
            metadata={
                "method": "writer_only",
                "word_count": len(full_content.split()),
                "sections_generated": len(sections_content),
                "knowledge_source": "llm_internal_only"
            }
        )
    
    def _generate_introduction(self, topic: str, main_headings: List[str]) -> str:
        """Generate an engaging introduction that previews the article structure."""
        
        headings_preview = ", ".join(main_headings[:4])  # Preview first 4 headings
        
        intro_prompt = f"""
        Write an engaging introduction (approximately 150-200 words) for an article about {topic}.
        
        The introduction should:
        1. Hook the reader with an interesting opening
        2. Provide essential context about why {topic} matters
        3. Clearly define what {topic} is for readers unfamiliar with it
        4. Preview the main areas that will be covered: {headings_preview}
        5. Set expectations for what readers will learn
        
        Write in an informative yet accessible style that would engage both
        newcomers and those already familiar with {topic}.
        """
        
        return self.call_api(intro_prompt).strip()
    
    def _generate_section_content(self, heading: str, subtopics: List[str], 
                                topic: str, article_title: str) -> str:
        """
        Generate detailed content for a specific section.
        
        This method tests the LLM's ability to develop ideas systematically
        and provide comprehensive coverage of specific aspects.
        """
        
        subtopics_text = "\n".join(f"- {subtopic}" for subtopic in subtopics)
        
        section_prompt = f"""
        Write a comprehensive section titled "{heading}" for an article about {topic}.
        
        This section should cover these specific subtopics:
        {subtopics_text}
        
        Requirements:
        1. Write approximately {self.words_per_section} words
        2. Organize content into 2-3 well-structured paragraphs
        3. Include specific examples and explanations where appropriate
        4. Maintain an informative, professional tone
        5. Ensure content flows logically and connects to the broader topic of {topic}
        6. Provide depth while remaining accessible to general readers
        
        Focus on delivering valuable, accurate information that demonstrates
        comprehensive understanding of this aspect of {topic}.
        """
        
        content = self.call_api(section_prompt).strip()
        
        # Ensure we have meaningful content even if generation fails
        if not content or len(content.split()) < 50:
            logger.warning(f"Generated content for section '{heading}' was too short, providing fallback")
            content = (f"This section explores {heading.lower()} in the context of {topic}. "
                      f"Understanding these concepts is crucial for grasping the full scope "
                      f"and implications of {topic} in both theoretical and practical applications.")
        
        return content
    
    def _generate_conclusion(self, topic: str, main_headings: List[str]) -> str:
        """Generate a conclusion that synthesizes the main points."""
        
        conclusion_prompt = f"""
        Write a thoughtful conclusion (approximately 150-200 words) for an article about {topic}.
        
        The article covered these main areas: {', '.join(main_headings)}
        
        The conclusion should:
        1. Synthesize the key insights from across all sections
        2. Emphasize the broader significance and relevance of {topic}
        3. Connect back to themes introduced in the opening
        4. Leave readers with a clear understanding of why {topic} matters
        5. Provide a sense of closure while inspiring further interest
        
        Avoid simply summarizing each section; instead, weave together the
        main themes to provide a cohesive final perspective on {topic}.
        """
        
        return self.call_api(conclusion_prompt).strip()