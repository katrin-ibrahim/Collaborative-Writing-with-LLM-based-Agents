# src/collaborative/agents/reviewer_agent.py
"""
Refactored ReviewerAgent using clean architecture with real tools only.
"""

import logging
import re
from typing import Any, Dict, List

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    fact_checking_strategy_prompt,
    qualitative_feedback_prompt,
)
from src.collaborative.data_models import ReviewFeedback
from src.collaborative.tools.reviewer_toolkit import ReviewerToolkit
from src.utils.data import Article

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """
    Reviewer agent that analyzes articles using objective metrics and fact-checking.

    Uses tools for objective analysis and LLM for qualitative assessment.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration
        self.max_claims_to_check = config.get("reviewer.max_claims_to_check", 5)
        self.max_search_results = config.get("reviewer.max_search_results", 3)
        self.rm_type = config.get("retrieval_manager_type", "wiki")

        # Initialize reviewer toolkit
        self.toolkit = ReviewerToolkit(config)

        # Get available tools
        self.search_tool = None
        self.metrics_tool = None

        for tool in self.toolkit.get_available_tools():
            if tool.name == "search_and_retrieve":
                self.search_tool = tool
            elif tool.name == "get_article_metrics":
                self.metrics_tool = tool

        logger.info("ReviewerAgent initialized with clean architecture")

    def process(self, article: Article) -> ReviewFeedback:
        """
        Review an article and return structured feedback.

        Args:
            article: Article to review

        Returns:
            ReviewFeedback with objective score and qualitative insights
        """
        logger.info(f"Reviewing article: {article.title}")

        try:
            # Step 1: Get objective metrics using tool
            metrics = self._get_article_metrics(article)

            # Step 2: Identify claims for fact-checking using LLM
            potential_claims = self._identify_claims_for_fact_checking(article)

            # Step 3: Fact-check important claims using search tool
            fact_check_results = self._fact_check_claims(potential_claims)

            # Step 4: Generate qualitative feedback using LLM
            qualitative_feedback = self._generate_qualitative_feedback(
                article, metrics, potential_claims, fact_check_results
            )

            # Step 5: Parse feedback into structured components
            issues, recommendations = self._parse_qualitative_feedback(
                qualitative_feedback
            )

            # Step 6: Calculate overall score based on metrics and fact-checking
            overall_score = self._calculate_overall_score(metrics, fact_check_results)

            # Step 7: Create structured feedback
            feedback = ReviewFeedback(
                overall_score=overall_score,
                feedback_text=qualitative_feedback,
                issues_count=len(issues),
                recommendations=recommendations,
            )

            logger.info(
                f"Review completed for {article.title}: score {feedback.overall_score:.3f}"
            )
            return feedback

        except Exception as e:
            logger.error(f"Review failed for {article.title}: {e}")
            # Return error feedback
            return ReviewFeedback(
                overall_score=0.0,
                feedback_text=f"Review failed due to technical error: {e}",
                issues_count=1,
                recommendations=["Fix technical issues preventing proper review"],
            )

    def _get_article_metrics(self, article: Article) -> Dict[str, Any]:
        """Get objective metrics using the metrics tool."""

        try:
            if self.metrics_tool:
                metrics = self.metrics_tool.invoke(
                    {"content": article.content, "title": article.title}
                )

                if metrics.get("analysis_success", False):
                    logger.info(
                        f"Metrics calculated: {metrics.get('word_count', 0)} words, {metrics.get('heading_count', 0)} headings"
                    )
                    return metrics
                else:
                    logger.warning(
                        f"Metrics calculation failed: {metrics.get('error', 'Unknown error')}"
                    )

            # Fallback to basic metrics if tool fails
            return self._calculate_basic_metrics(article)

        except Exception as e:
            logger.warning(f"Metrics tool failed: {e}, using fallback")
            return self._calculate_basic_metrics(article)

    def _calculate_basic_metrics(self, article: Article) -> Dict[str, Any]:
        """Fallback basic metrics calculation."""
        content = article.content

        return {
            "title": article.title,
            "word_count": len(content.split()),
            "character_count": len(content),
            "heading_count": len(re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)),
            "headings": re.findall(r"^#+\s+(.+)$", content, re.MULTILINE),
            "paragraph_count": len([p for p in content.split("\n\n") if p.strip()]),
            "analysis_success": True,
        }

    def _identify_claims_for_fact_checking(self, article: Article) -> List[str]:
        """Use LLM to identify important claims that need fact-checking."""

        try:
            # Use template prompt to identify claims
            prompt = fact_checking_strategy_prompt(article.content, article.title)
            response = self.api_client.call_api(prompt)

            # Parse numbered list from response
            claims = []
            for line in response.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Extract claim text after number/bullet
                    claim_text = re.sub(r"^[\d\-\.\)\s]+", "", line).strip()
                    if claim_text and len(claim_text) > 10:  # Reasonable length check
                        claims.append(claim_text)

            # Limit to configured maximum
            claims = claims[: self.max_claims_to_check]
            logger.info(f"Identified {len(claims)} claims for fact-checking")
            return claims

        except Exception as e:
            logger.warning(f"Claim identification failed: {e}")
            # Fallback to simple pattern-based identification
            return self._fallback_claim_identification(article.content)

    def _fallback_claim_identification(self, content: str) -> List[str]:
        """Simple fallback for claim identification using patterns."""

        sentences = re.split(r"[.!?]+", content)
        potential_claims = []

        # Simple patterns for factual-looking statements
        factual_patterns = [
            r"\b\d{4}\b",  # Years
            r"\b\d+%\b",  # Percentages
            r"\b\d+\s*(million|billion|thousand)\b",  # Large numbers
            r"\baccording\s+to\b",  # Attribution
            r"\bresearch\s+shows?\b",  # Research claims
        ]

        for sentence in sentences[:20]:  # Check first 20 sentences
            sentence = sentence.strip()
            if len(sentence) > 30:  # Skip very short sentences
                for pattern in factual_patterns:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        potential_claims.append(sentence)
                        break

        return potential_claims[: self.max_claims_to_check]

    def _fact_check_claims(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Fact-check claims using search tool."""

        fact_check_results = []

        for claim in claims:
            try:
                if self.search_tool:
                    # Search for verification of the claim
                    search_result = self.search_tool.invoke(
                        {
                            "query": claim,
                            "rm_type": self.rm_type,
                            "max_results": self.max_search_results,
                            "purpose": "fact_checking",
                        }
                    )

                    sources_found = (
                        search_result.get("total_results", 0)
                        if search_result.get("success")
                        else 0
                    )

                    fact_check_results.append(
                        {
                            "claim": claim,
                            "sources_found": sources_found,
                            "search_successful": search_result.get("success", False),
                            "search_results": search_result.get("results", [])[
                                :2
                            ],  # Keep top 2 results
                        }
                    )

                    logger.debug(
                        f"Fact-checked claim: '{claim[:50]}...' → {sources_found} sources"
                    )

            except Exception as e:
                logger.warning(f"Fact-check failed for claim '{claim[:50]}...': {e}")
                fact_check_results.append(
                    {
                        "claim": claim,
                        "sources_found": 0,
                        "search_successful": False,
                        "error": str(e),
                    }
                )

        logger.info(
            f"Fact-checking completed: {len(fact_check_results)} claims processed"
        )
        return fact_check_results

    def _generate_qualitative_feedback(
        self,
        article: Article,
        metrics: Dict[str, Any],
        potential_claims: List[str],
        fact_check_results: List[Dict[str, Any]],
    ) -> str:
        """Generate qualitative feedback using LLM based on analysis results."""

        try:
            # Use template prompt for qualitative feedback
            prompt = qualitative_feedback_prompt(
                article_title=article.title,
                article_content=article.content,
                metrics=metrics,
                potential_claims=potential_claims,
                fact_check_results=fact_check_results,
            )

            feedback = self.api_client.call_api(prompt)
            return feedback

        except Exception as e:
            logger.warning(f"LLM feedback generation failed: {e}")
            # Fallback feedback based on metrics
            return self._generate_fallback_feedback(metrics, fact_check_results)

    def _generate_fallback_feedback(
        self, metrics: Dict[str, Any], fact_check_results: List[Dict[str, Any]]
    ) -> str:
        """Generate basic feedback when LLM fails."""

        word_count = metrics.get("word_count", 0)
        heading_count = metrics.get("heading_count", 0)

        issues = []
        recommendations = []

        if word_count < 500:
            issues.append("Article appears to be quite short")
            recommendations.append(
                "Consider expanding the content with more detailed information"
            )

        if heading_count == 0:
            issues.append("No headings found - article lacks structure")
            recommendations.append("Add section headings to improve organization")

        verified_claims = sum(
            1 for result in fact_check_results if result.get("sources_found", 0) > 0
        )
        total_claims = len(fact_check_results)

        if total_claims > 0 and verified_claims < total_claims // 2:
            issues.append("Several claims could not be verified with sources")
            recommendations.append("Add citations or verify factual claims")

        feedback = "MAIN ISSUES:\n" + "\n".join([f"- {issue}" for issue in issues])
        feedback += "\n\nRECOMMENDATIONS:\n" + "\n".join(
            [f"- {rec}" for rec in recommendations]
        )
        feedback += f"\n\nDETAILED FEEDBACK:\nBasic analysis completed. Article has {word_count} words and {heading_count} headings."

        return feedback

    def _calculate_overall_score(
        self, metrics: Dict[str, Any], fact_check_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall score based on metrics and fact-checking."""

        score = 0.5  # Base score

        # Word count component (0-0.3)
        word_count = metrics.get("word_count", 0)
        if word_count >= 1000:
            score += 0.3
        elif word_count >= 500:
            score += 0.2
        elif word_count >= 200:
            score += 0.1

        # Structure component (0-0.2)
        heading_count = metrics.get("heading_count", 0)
        if heading_count >= 4:
            score += 0.2
        elif heading_count >= 2:
            score += 0.1

        # Fact-checking component (0-0.3)
        if fact_check_results:
            verified_claims = sum(
                1 for result in fact_check_results if result.get("sources_found", 0) > 0
            )
            verification_rate = verified_claims / len(fact_check_results)
            score += verification_rate * 0.3
        else:
            # No claims to verify - neutral
            score += 0.15

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

    def _parse_qualitative_feedback(
        self, feedback_text: str
    ) -> tuple[List[str], List[str]]:
        """Parse qualitative feedback into structured issues and recommendations."""

        issues = self._extract_section(feedback_text, "MAIN ISSUES:")
        recommendations = self._extract_section(feedback_text, "RECOMMENDATIONS:")

        # Ensure we have at least some feedback
        if not issues:
            issues = ["No specific issues identified"]
        if not recommendations:
            recommendations = ["Continue with current approach"]

        return issues, recommendations

    def _extract_section(self, text: str, section_header: str) -> List[str]:
        """Extract bulleted items from a section."""

        # Find section start
        section_pattern = re.escape(section_header) + r"\s*(.*?)(?=\n[A-Z]+:|$)"
        section_match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)

        if not section_match:
            return []

        section_content = section_match.group(1).strip()

        # Extract bullet points
        items = []
        for line in section_content.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("•"):
                item = line[1:].strip()
                if item:
                    items.append(item)

        # If no bullet points found, return the whole content as single item
        if not items and section_content:
            items = [section_content]

        return items
