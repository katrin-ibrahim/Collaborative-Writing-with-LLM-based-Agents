from typing import Dict
import logging
from .base_workflow import BaseWorkflow
from ..utils.data_models import Article

logger = logging.getLogger(__name__)

class DirectPromptingWorkflow(BaseWorkflow):
    """
    Baseline 1: Direct Prompting
    One-shot generation without outline or retrieval.
    """
    
    def generate_content(self, topic: str) -> Article:
        """Generate article using single direct prompt."""
        
        logger.info(f"Generating content via direct prompting for: {topic}")
        
        # Create comprehensive prompt for one-shot generation
        prompt = f"""
        Write a comprehensive, well-structured article about: {topic}
        
        Requirements:
        1. Include a clear title
        2. Organize content with headers and subheaders
        3. Provide detailed explanations and examples
        4. Ensure logical flow and coherent structure
        5. Aim for 800-1200 words
        
        Write the complete article now:
        """
        
        # Generate content in one shot
        content = self.call_api(prompt)
        
        # Parse the response to extract title and structure
        article = self._parse_direct_response(content, topic)
        
        logger.info(f"Direct prompting completed: {len(article.content)} characters")
        return article
    
    def _parse_direct_response(self, response: str, topic: str) -> Article:
        """Parse the direct response into structured Article format."""
        
        lines = response.strip().split('\n')
        title = topic  # Default title
        content = response
        sections = {}
        
        # Try to extract title from first line
        if lines and (lines[0].startswith('#') or 'title' in lines[0].lower()):
            title = lines[0].replace('#', '').replace('Title:', '').strip()
        
        # Simple section parsing
        current_section = None
        section_content = []
        
        for line in lines:
            if line.strip().startswith('##'):
                if current_section:
                    sections[current_section] = '\n'.join(section_content)
                current_section = line.replace('##', '').strip()
                section_content = []
            elif current_section:
                section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return Article(
            title=title,
            content=content,
            sections=sections,
            metadata={"method": "direct_prompting", "word_count": len(content.split())}
        )