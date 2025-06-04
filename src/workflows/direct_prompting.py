# src/workflows/direct_prompting.py
from typing import Dict
import logging
import os
from workflows.base_workflow import BaseWorkflow
from utils.data_models import Article
from utils.api import APIClient

logger = logging.getLogger(__name__)

class DirectPromptingWorkflow(BaseWorkflow):
    """
    Direct Prompting Workflow with proper API integration.
    
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        api_token = os.getenv('HF_TOKEN') or os.getenv('API_TOKEN')
        self.api_client = APIClient(api_token=api_token)
        logger.info("DirectPromptingWorkflow initialized with real API client")
    
    def generate_content(self, topic: str) -> Article:
        """Generate article using single direct prompt with real API."""
        
        logger.info(f"Generating content via direct prompting for: {topic}")
        
        prompt = f"""
        Write a comprehensive, well-structured article about: {topic}

        Requirements:
        1. Start with a clear title using # markdown syntax
        2. Include 4-6 main sections with ## headings
        3. Each section should be 200-300 words
        4. Include specific details, examples, and explanations
        5. Use proper markdown formatting
        6. Target 1000-1500 words total
        7. Ensure logical flow between sections

        Structure should include:
        - Introduction/Overview
        - Background/History (if applicable)  
        - Key concepts or components
        - Current applications or examples
        - Challenges or considerations
        - Future outlook or conclusion

        Write the complete article now with proper markdown formatting:
        """
        
        try:
            content = self.api_client.call_api(prompt)
            logger.info(f"API returned {len(content)} characters")
        except Exception as e:
            logger.error(f"API call failed: {e}")
            content = f"# {topic}\n\nError generating content: {e}"
        
        # Parse the response to extract title and structure
        article = self._parse_direct_response(content, topic)
        
        logger.info(f"Direct prompting completed: {len(article.content)} characters")
        return article
    
    def _parse_direct_response(self, response: str, topic: str) -> Article:
        """Enhanced parsing of the direct response into structured Article format."""
        
        lines = response.strip().split('\n')
        title = topic  # Default title
        content = response
        sections = {}
        
        # Extract title from first line if it starts with #
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        # Enhanced section parsing
        current_section = None
        section_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for main headings (##)
            if line_stripped.startswith('## '):
                # Save previous section
                if current_section and section_content:
                    sections[current_section] = '\n'.join(section_content).strip()
                
                # Start new section
                current_section = line_stripped[3:].strip()
                section_content = []
                
            elif current_section:
                # Add content to current section
                section_content.append(line)
        
        # Add final section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content).strip()
        
        # Calculate word count
        word_count = len(content.split())
        
        return Article(
            title=title,
            content=content,
            sections=sections,
            metadata={
                "method": "direct_prompting",
                "word_count": word_count,
                "section_count": len(sections),
                "api_used": "real_api_client"
            }
        )