from typing import Dict
import logging
from workflows.base_workflow import BaseWorkflow
from agents.writer.writer_agent import WriterAgent
from utils.data_models import Article

logger = logging.getLogger(__name__)

class RAGWriterWorkflow(BaseWorkflow):
    """
    Baseline: RAG + Writer workflow using full knowledge capabilities.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Configure WriterAgent for full knowledge capabilities
        writer_config = config.copy()
        writer_config['writer.use_external_knowledge'] = True
        writer_config['writer.use_knowledge_base'] = True
        writer_config['writer.knowledge_depth'] = 'semantic_hierarchical'
        
        self.writer_agent = WriterAgent(writer_config)
    
    def generate_content(self, topic: str) -> Article:
        """Generate content using WriterAgent with full knowledge capabilities."""
        logger.info(f"Generating content via RAG+Writer workflow for: {topic}")
        
        article = self.writer_agent.process(topic)
        
        # Update metadata to reflect workflow approach
        article.metadata["workflow"] = "rag_writer"
        article.metadata["knowledge_approach"] = "external_retrieval_organized"
        
        logger.info(f"RAG+Writer workflow completed: {len(article.content)} characters")
        return article