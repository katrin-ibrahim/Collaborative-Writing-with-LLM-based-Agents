# src/methods/writer_method.py
"""
Writer-only method using sophisticated 3-node WriterAgent workflow.
"""

import logging
from typing import Any, Dict

from src.collaborative.agents.writer_agent import WriterAgent
from src.methods.base_method import BaseMethod
from src.utils.data import Article

logger = logging.getLogger(__name__)


class WriterMethod(BaseMethod):
    """
    Single writer agent using sophisticated planning-first workflow.

    Uses the existing 3-node WriterAgent:
    plan_outline → targeted_research → refine_outline → write_content
    """

    def __init__(self, client, config: Dict[str, Any]):
        super().__init__(client, config)

        # Writer configuration
        self.writer_config = {
            **config,
            "writer.max_research_iterations": config.get(
                "writer.max_research_iterations", 2
            ),
            "writer.use_external_knowledge": config.get(
                "writer.use_external_knowledge", True
            ),
        }

        logger.info("WriterMethod initialized with 3-node workflow")

    def run(self, topic: str) -> Article:
        """
        Generate article using sophisticated writer agent.

        Args:
            topic: Topic to write about

        Returns:
            Generated article with writer metadata
        """
        logger.info(f"Running writer method for: {topic}")

        try:
            # Initialize writer agent
            writer = WriterAgent(self.writer_config)

            # Generate article using 3-node workflow
            article = writer.process(topic)

            # Update metadata to indicate method used
            article.metadata.update(
                {
                    "method": self.get_method_name(),
                    "agent_type": "sophisticated_writer",
                    "workflow_nodes": 3,
                }
            )

            logger.info(f"Writer method completed for {topic}")
            return article

        except Exception as e:
            logger.error(f"Writer method failed for '{topic}': {e}")
            # Return error article
            from src.utils.article import error_article

            return error_article(topic, str(e), self.get_method_name())
