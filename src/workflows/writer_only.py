# src/workflows/writer_only.py 
from typing import Dict, List
import logging

from agents.writer.writer_agent import WriterAgent
from workflows.base_workflow import BaseWorkflow
from utils.data_models import Article


logger = logging.getLogger(__name__)

class WriterOnlyWorkflow(BaseWorkflow):
    """
    Baseline: Writer-Only workflow using internal knowledge only.
    
    This workflow tests the hypothesis: "How well can structured writing
    work using only the language model's internal knowledge?"
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Configure WriterAgent for internal-knowledge-only operation
        writer_config = config.copy()
        writer_config['writer.use_external_knowledge'] = False
        writer_config['writer.use_knowledge_base'] = False
        
        self.writer_agent = WriterAgent(writer_config)
    
    def generate_content(self, topic: str) -> Article:
        """Generate content using WriterAgent in internal-knowledge-only mode."""
        logger.info(f"Generating content via writer-only workflow for: {topic}")
        
        article = self.writer_agent.process(topic)
        
        # Update metadata to reflect workflow constraints
        article.metadata["workflow"] = "writer_only"
        article.metadata["knowledge_source"] = "llm_internal_only"
        
        logger.info(f"Writer-only workflow completed: {len(article.content)} characters")
        return article