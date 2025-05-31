# src/agents/writer/outline_generator.py
import os
from typing import Dict, Any
import logging
from utils.api import APIClient
from utils.data_models import Outline

logger = logging.getLogger(__name__)

class OutlineGenerator:
    """Utility for generating structured article outlines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get API client - same one used throughout the system
        api_token = os.getenv('HF_TOKEN') or os.getenv('API_TOKEN')
        self.api_client = APIClient(api_token=api_token)
    
    def generate_outline(self, topic: str, context: str = "") -> Outline:
        """Generate a hierarchical outline for the given topic."""
        
        outline_prompt = f"""
        Generate a detailed outline for an article about: {topic}
        
        {'Based on the following context:' + context if context else ''}
        
        Create a hierarchical outline with:
        1. A clear title
        2. 4-6 main headings
        3. 2-3 subheadings under each main heading
        
        Format the response as:
        Title: [Article Title]
        
        1. [Main Heading 1]
           - [Subheading 1.1]
           - [Subheading 1.2]
        2. [Main Heading 2]
           - [Subheading 2.1]
           - [Subheading 2.2]
        ...
        """
        
        # Use the API client directly
        response = self.api_client.call_api(outline_prompt)
        
        # Parse the response into structured outline
        outline = self._parse_outline_response(response, topic)
        
        self.logger.info(f"Generated outline with {len(outline.headings)} main headings")
        return outline
    
    def _parse_outline_response(self, response: str, topic: str) -> Outline:
        """Parse the model response into a structured Outline object."""
        lines = response.strip().split('\n')
        
        # Extract title (fallback to topic if not found)
        title = topic
        headings = []
        subheadings = {}
        current_heading = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                title = line.replace('Title:', '').strip()
            elif line and line[0].isdigit() and '.' in line:
                # Main heading
                heading = line.split('.', 1)[1].strip()
                headings.append(heading)
                current_heading = heading
                subheadings[heading] = []
            elif line.startswith('-') and current_heading:
                # Subheading
                subheading = line.replace('-', '').strip()
                subheadings[current_heading].append(subheading)
        
        # Fallback outline if parsing fails
        if not headings:
            headings = [
                "Introduction",
                "Background and Context", 
                "Key Concepts",
                "Current Developments",
                "Future Implications",
                "Conclusion"
            ]
            subheadings = {heading: [f"{heading} - Detail 1", f"{heading} - Detail 2"] 
                          for heading in headings}
        
        return Outline(title=title, headings=headings, subheadings=subheadings)
