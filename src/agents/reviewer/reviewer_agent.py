import logging
from typing import Any, Dict

from src.agents.base_agent import BaseAgent
from src.agents.tools.agent_toolkit import AgentToolkit
from src.utils.data.models import Article

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """
    Simple Reviewer agent that provides basic article review.
    Uses real tools for claim extraction and LLM for analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Initialize real tools
        self.toolkit = AgentToolkit(config)

        logger.info("Simple ReviewerAgent initialized")

    def review(self, article: Article) -> str:
        """
        Review an article and return feedback.

        Args:
            article: Article to review

        Returns:
            String with review feedback
        """
        logger.info(f"Reviewing article: {article.title}")

        try:
            # 1. Extract claims using real tool
            claims_result = self.toolkit.extract_content_claims(
                content=article.content, focus_types=["factual", "statistical"]
            )

            # 2. Basic LLM review
            review_prompt = f"""
            Review this article and provide constructive feedback:

            Title: {article.title}
            Content: {article.content}

            Claims extracted: {len(claims_result['claims'])} factual claims found
            Key claims: {[claim['text'] for claim in claims_result['claims'][:3]]}

            Please provide:
            1. Overall assessment (Good/Fair/Needs Improvement)
            2. Main strengths
            3. Main issues or concerns
            4. Specific recommendations for improvement
            5. Focus on factual accuracy, clarity, and completeness

            Keep the review constructive and specific.
            """

            review = self.api_client.call_api(review_prompt)

            logger.info(f"Review completed for: {article.title}")
            return review

        except Exception as e:
            logger.error(f"Review failed for {article.title}: {e}")
            return f"Review failed: {e}"
