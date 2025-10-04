# src/methods/writer_method.py
"""
Writer-only method using sophisticated 3-node WriterAgent workflow.
"""

import logging

from src.collaborative.agents.writer_agent import WriterAgent
from src.collaborative.memory.memory import SharedMemory
from src.config.config_context import ConfigContext
from src.methods.base_method import BaseMethod
from src.utils.data import Article

logger = logging.getLogger(__name__)


class WriterMethod(BaseMethod):
    """
    Single writer agent using sophisticated planning-first workflow.

    Uses the existing 3-node WriterAgent:
    plan_outline → targeted_research → refine_outline → write_content
    """

    def __init__(self):
        super().__init__()
        self.collab_config = ConfigContext.get_collaboration_config()

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
            memory = SharedMemory(
                topic=topic,
                max_iterations=self.collab_config.max_iterations,
                min_feedback_threshold=0,
                tom_enabled=False,  # Writer-only method doesn't use ToM
            )
            ConfigContext.set_memory_instance(memory)

            writer = WriterAgent()

            seed_article = Article(
                title=topic,
                content=f"# {topic}\n\n",
                sections={},
                metadata=memory.state.get("metadata", {}).copy(),
            )
            memory.update_article_state(seed_article)

            writer.process()

            final_article = Article(
                title=memory.state.get("topic", topic),
                content=memory.state.get("article_content", ""),
                sections=memory.state.get("article_sections_by_iteration", {}).get(
                    str(memory.get_current_iteration()), {}
                ),
                metadata=memory.state.get("metadata", {}),
            )

            final_article.metadata.update(
                {
                    "method": self.get_method_name(),
                    "agent_type": "writer_only",
                }
            )

            logger.info(f"Writer method completed for {topic}")
            return final_article

        except Exception as e:
            logger.error(f"Writer method failed for '{topic}': {e}")
            # Return error article
            from src.utils.article import error_article

            return error_article(topic, str(e), self.get_method_name())
