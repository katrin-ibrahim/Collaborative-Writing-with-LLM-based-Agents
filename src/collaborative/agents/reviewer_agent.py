# src/collaborative/agents/reviewer_agent.py
"""
Refactored ReviewerAgent using clean architecture with real tools only.
"""

import time

import logging
import re
from typing import List

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    agentic_review_prompt,
    enhanced_feedback_prompt,
    fact_checking_prompt,
    feedback_prompt,
    reviewer_search_strategy_prompt,
    reviewer_tool_decision_prompt,
)
from src.collaborative.tools.reviewer_toolkit import ReviewerToolkit
from src.config.config_context import ConfigContext
from src.utils.data import Article, ArticleMetrics, FactCheckResult

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """
    Reviewer agent that analyzes articles using objective metrics and fact-checking.

    Uses tools for objective analysis and LLM for qualitative assessment.
    """

    def __init__(self):
        super().__init__()
        self.collaboration_config = ConfigContext.get_collaboration_config()
        self.retrieval_config = ConfigContext.get_retrieval_config()

        # Configuration
        self.max_claims_to_check = getattr(
            self.collaboration_config, "reviewer.max_claims_to_check", 10
        )
        self.max_search_results = getattr(
            self.collaboration_config, "reviewer.max_search_results", 3
        )
        self.rm_type = getattr(self.retrieval_config, "retrieval_manager", "wiki")

        # Initialize reviewer toolkit
        self.toolkit = ReviewerToolkit(self.retrieval_config)

        # Get available tools
        self.metrics_tool = None
        self.verify_claims_tool = None

        for tool in self.toolkit.get_available_tools():
            if tool.name == "get_article_metrics":
                self.metrics_tool = tool
            elif tool.name == "verify_claims_with_research":
                self.verify_claims_tool = tool

        logger.info("ReviewerAgent initialized")

    def process(self) -> None:
        """
        HYBRID AGENTIC: Review article with forced fact-checking + agent tool decisions.
        """
        # Get shared memory from ConfigContext
        shared_memory = ConfigContext.get_memory_instance()
        if not shared_memory:
            raise RuntimeError(
                "SharedMemory not available in ConfigContext. Make sure it's initialized before calling ReviewerAgent.process()"
            )

        # Create article object from shared memory for analysis
        article = Article(
            title=shared_memory.state.topic,
            content=shared_memory.state.article_content,
            sections=shared_memory.state.article_sections_by_iteration.get(
                str(shared_memory.state.iteration), {}
            ),
            metadata=shared_memory.state.metadata,
        )

        logger.info(f"Reviewing article with hybrid agentic approach: {article.title}")

        # FORCED: Always get metrics and extract claims (essential steps)
        metrics = self._get_article_metrics(article)
        potential_claims = self._extract_claims_for_fact_checking(article)

        # AGENTIC: Let agent decide which tools to use for comprehensive review
        available_tools = [tool.name for tool in self.toolkit.get_available_tools()]
        unused_tools = [
            name
            for name in available_tools
            if name not in ["verify_claims_with_research"]
        ]

        tool_decision_prompt_text = reviewer_tool_decision_prompt(
            article.title, unused_tools
        )
        tool_decision = self.api_client.call_api(tool_decision_prompt_text)

        # Parse agent's tool decisions
        selected_tools = []
        tool_reasoning = ""
        for line in tool_decision.split("\n"):
            if line.startswith("REVIEW_TOOLS:"):
                tools_str = line.replace("REVIEW_TOOLS:", "").strip()
                selected_tools = [
                    tool.strip() for tool in tools_str.split(",") if tool.strip()
                ]
            elif line.startswith("REASONING:"):
                tool_reasoning = line.replace("REASONING:", "").strip()

        logger.info(f"Agent selected review tools: {selected_tools}")

        # FORCED: Always do fact-checking (essential for review quality)
        fact_check_results = self._fact_check_claims(potential_claims)

        # AGENTIC: Execute agent's selected supplementary tools
        supplementary_context = self._execute_agentic_review_tools(
            selected_tools, article
        )

        # FORCED: Always generate feedback (but enhanced with agentic context)
        qualitative_feedback = self._generate_agentic_informed_feedback(
            article,
            metrics,
            potential_claims,
            fact_check_results,
            supplementary_context,
        )

        # Parse and score
        issues, recommendations = self._parse_qualitative_feedback(qualitative_feedback)
        overall_score = self._calculate_overall_score(metrics, fact_check_results)

        logger.info(
            f"Hybrid agentic review completed: score {overall_score:.3f}, tools used: {selected_tools}"
        )

        # Store feedback directly in shared memory using the proper method
        shared_memory.store_structured_feedback(
            feedback_text=qualitative_feedback,
            iteration=shared_memory.get_current_iteration(),
            target_section=None,
        )

        # Also store additional metadata in the memory state
        feedback_metadata = {
            "overall_score": overall_score,
            "issues": issues,
            "recommendations": recommendations,
            "metrics": metrics.to_dict() if hasattr(metrics, "to_dict") else metrics,
            "fact_check_results": [
                result.to_dict() if hasattr(result, "to_dict") else result
                for result in fact_check_results
            ],
            "tools_used": selected_tools,
            "timestamp": time.time(),
        }

        # Store in metadata for easy access
        if "review_results" not in shared_memory.state.metadata:
            shared_memory.state.metadata["review_results"] = []
        shared_memory.state.metadata["review_results"].append(feedback_metadata)

    def _execute_agentic_review_tools(
        self, selected_tools: List[str], article: Article
    ) -> str:
        """Execute the tools the agent selected for comprehensive review."""
        tool_map = {tool.name: tool for tool in self.toolkit.get_available_tools()}
        context_parts = []

        for tool_name in selected_tools:
            if tool_name in tool_map:
                try:
                    if tool_name == "search_and_retrieve":
                        # Let agent decide what to search for
                        potential_claims = self._extract_claims_for_fact_checking(
                            article
                        )
                        search_strategy_prompt_text = reviewer_search_strategy_prompt(
                            article.title, [claim for claim in potential_claims[:5]]
                        )
                        search_decision = self.api_client.call_api(
                            search_strategy_prompt_text
                        )

                        # Parse search queries
                        search_queries = []
                        for line in search_decision.split("\n"):
                            if line.startswith("SEARCH_QUERIES:"):
                                queries_str = line.replace(
                                    "SEARCH_QUERIES:", ""
                                ).strip()
                                search_queries = [
                                    q.strip()
                                    for q in queries_str.split(",")
                                    if q.strip()
                                ]
                                break

                        # Execute searches
                        for query in search_queries[:2]:  # Limit to 2 searches
                            result = tool_map[tool_name].invoke(query)
                            if result.get("success"):
                                context_parts.append(
                                    f"SEARCH RESULT for '{query}': Found {result.get('total_chunks', 0)} sources"
                                )
                                logger.info(
                                    f"Agent search '{query}': {result.get('total_chunks', 0)} chunks"
                                )

                    elif tool_name == "get_chunks_by_ids":
                        # Agent decides which chunks to retrieve for detailed verification
                        memory = ConfigContext.get_memory_instance()
                        if memory:
                            chunk_summaries = memory.get_chunk_summaries()
                            if chunk_summaries:
                                # Let agent select relevant chunks
                                chunk_list = list(chunk_summaries.keys())[
                                    :10
                                ]  # Limit options
                                if chunk_list:
                                    chunk_selection = (
                                        f"{chunk_list[0]},{chunk_list[1]}"
                                        if len(chunk_list) > 1
                                        else chunk_list[0]
                                    )
                                    result = tool_map[tool_name].invoke(
                                        {"chunk_ids": chunk_selection.split(",")}
                                    )
                                    if result.get("success"):
                                        context_parts.append(
                                            f"RETRIEVED CHUNKS: {result.get('retrieved_count', 0)} detailed sources for verification"
                                        )
                                        logger.info(
                                            f"Agent retrieved {result.get('retrieved_count', 0)} chunks"
                                        )

                    elif tool_name == "get_current_iteration":
                        result = tool_map[tool_name].invoke({})
                        if result.get("success"):
                            context_parts.append(
                                f"ITERATION CONTEXT: {result.get('message', '')}"
                            )
                            logger.info("Agent used iteration context")

                    elif tool_name == "get_feedback":
                        result = tool_map[tool_name].invoke({"only_pending": True})
                        if result.get("success"):
                            context_parts.append(
                                f"EXISTING FEEDBACK: {result.get('feedback_summary', '')}"
                            )
                            logger.info("Agent checked existing feedback")

                except Exception as e:
                    logger.warning(f"Agentic tool {tool_name} failed: {e}")

        return (
            "\n".join(context_parts)
            if context_parts
            else "No additional context retrieved"
        )

    def _generate_agentic_informed_feedback(
        self,
        article: Article,
        metrics: ArticleMetrics,
        potential_claims: List[str],
        fact_check_results: List[FactCheckResult],
        supplementary_context: str,
    ) -> str:
        """Generate feedback enhanced with agent's supplementary context."""

        # Use the original feedback prompt but enhance with agentic context
        base_feedback = feedback_prompt(
            article.title,
            article.content,
            metrics,
            potential_claims,
            fact_check_results,
        )

        # Add agentic enhancement
        enhanced_prompt = enhanced_feedback_prompt(base_feedback, supplementary_context)

        return self.api_client.call_api(enhanced_prompt)

    def _get_article_metrics(self, article: Article) -> ArticleMetrics:
        """Get objective metrics using the metrics tool."""

        try:
            if self.metrics_tool:
                metrics_dict = self.metrics_tool.invoke(
                    {"content": article.content, "title": article.title}
                )

                if metrics_dict.get("analysis_success", False):
                    logger.info(
                        f"Metrics calculated: {metrics_dict.get('word_count', 0)} words, {metrics_dict.get('heading_count', 0)} headings"
                    )
                    return ArticleMetrics(**metrics_dict)
                else:
                    logger.warning(
                        f"Metrics calculation failed: {metrics_dict.get('error', 'Unknown error')}"
                    )

            # Fallback to basic metrics if tool fails
            return self._calculate_basic_metrics(article)

        except Exception as e:
            logger.warning(f"Metrics tool failed: {e}, using fallback")
            return self._calculate_basic_metrics(article)

    def _calculate_basic_metrics(self, article: Article) -> ArticleMetrics:
        """Fallback basic metrics calculation."""
        content = article.content

        return ArticleMetrics(
            title=article.title,
            word_count=len(content.split()),
            character_count=len(content),
            heading_count=len(re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)),
            headings=re.findall(r"^#+\s+(.+)$", content, re.MULTILINE),
            paragraph_count=len([p for p in content.split("\n\n") if p.strip()]),
            analysis_success=True,
        )

    def _extract_claims_for_fact_checking(self, article: Article) -> List[str]:
        """
        Step 1: Extract factual claims from article that need verification.
        """
        try:
            prompt = fact_checking_prompt(
                article_content=article.content, title=article.title
            )

            response = self.api_client.call_api(prompt)

            # Debug: log actual response to understand format
            logger.debug(f"Raw claims extraction response: {response[:500]}...")

            claims = self._parse_claims_from_response(response, article.content)

            logger.info(f"Claims extraction completed: {len(claims)} claims identified")
            return claims

        except Exception as e:
            logger.warning(f"Claims extraction failed: {e}")
            # Fallback: extract directly from article content
            return self._extract_claims_directly_from_article(article.content)

    def _generate_informed_feedback(
        self,
        article: Article,
        metrics: ArticleMetrics,
        claims: List[str],
        fact_check_results: List[FactCheckResult],
    ) -> str:
        """
        Step 3: Generate iteration-aware feedback using memory tools for context.
        """
        try:
            # Get available tools for agentic decision making
            tools = {tool.name: tool for tool in self.toolkit.get_available_tools()}

            # Gather iteration context using memory tools
            context_info = []
            iteration_num = 0  # Default iteration

            # Get current iteration to adjust review strategy
            if "get_current_iteration" in tools:
                try:
                    iteration_info = tools["get_current_iteration"].invoke({})
                    context_info.append(f"ITERATION CONTEXT: {iteration_info}")
                    logger.info(f"Reviewer using iteration context: {iteration_info}")

                    # Parse iteration number for strategy adjustment
                    iteration_num = (
                        iteration_info.get("iteration", 0)
                        if isinstance(iteration_info, dict)
                        else 0
                    )
                    if iteration_num == 0:
                        review_focus = (
                            "structural organization and content completeness"
                        )
                    else:
                        review_focus = (
                            "refinement, accuracy, and addressing previous feedback"
                        )

                    context_info.append(
                        f"REVIEW FOCUS: Prioritize {review_focus} for iteration {iteration_num}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get iteration info: {e}")
                    review_focus = "comprehensive review"

            # Check if we can compare with previous versions
            if "get_section_from_iteration" in tools and iteration_num > 0:
                try:
                    # Example: Compare introduction sections if they exist
                    for heading in metrics.headings[
                        :2
                    ]:  # Check first couple of headings
                        prev_section = tools["get_section_from_iteration"].invoke(
                            {"section_name": heading, "iteration": iteration_num - 1}
                        )
                        if prev_section.get("success"):
                            context_info.append(
                                f"PREVIOUS VERSION of '{heading}': {prev_section}"
                            )
                            break  # Just get one comparison for context
                except Exception as e:
                    logger.warning(f"Failed to get section comparison: {e}")

            # Prepare fact-check context
            fact_check_summary = (
                f"Verified {sum(1 for r in fact_check_results if r.verified)} claims, "
                f"unverified {sum(1 for r in fact_check_results if not r.verified)} claims"
            )

            # Create context-aware review prompt
            context_text = (
                "\n".join(context_info)
                if context_info
                else "Initial review - no previous context."
            )

            agentic_prompt = agentic_review_prompt(
                article.title,
                context_text,
                {
                    "word_count": metrics.word_count,
                    "heading_count": metrics.heading_count,
                    "headings": metrics.headings,
                    "paragraph_count": metrics.paragraph_count,
                },
                fact_check_summary,
                article.content,
            )

            response = self.api_client.call_api(agentic_prompt)

            logger.info("Informed feedback generation completed")
            return response.strip()

        except Exception as e:
            logger.warning(f"Informed feedback generation failed: {e}")
            return self._generate_fallback_feedback(metrics, fact_check_results)

    def _generate_fallback_feedback(
        self, metrics: ArticleMetrics, fact_check_results: List[FactCheckResult]
    ) -> str:
        """
        Generate basic structured feedback when LLM calls fail.
        """
        # Count verified/unverified claims
        verified_count = sum(1 for r in fact_check_results if r.verified == True)
        unverified_count = sum(1 for r in fact_check_results if r.sources_found == 0)

        feedback_parts = []

        # Content Quality Issues
        issues = []
        if metrics.word_count < 500:
            issues.append("Article appears too short for comprehensive coverage")
        if metrics.heading_count == 0:
            issues.append("No section headings found - article lacks clear structure")
        if unverified_count > 0:
            issues.append(
                f"{unverified_count} claims could not be verified with available sources"
            )

        if issues:
            feedback_parts.append("## Content Quality Issues")
            feedback_parts.extend([f"- {issue}" for issue in issues])

        # Structural Assessment
        structure_notes = []
        if metrics.heading_count < 3:
            structure_notes.append(
                "Consider adding more section headings for better organization"
            )
        if metrics.paragraph_count < 5:
            structure_notes.append(
                "Article structure appears minimal - may need more detailed sections"
            )

        if structure_notes:
            feedback_parts.append("\n## Structural Assessment")
            feedback_parts.extend([f"- {note}" for note in structure_notes])

        # Improvement Recommendations
        recommendations = [
            "Add more detailed content to reach optimal article length",
            "Include section headings to improve readability",
        ]
        if unverified_count > 0:
            recommendations.append(
                "Verify claims with reliable sources or qualify uncertain statements"
            )

        feedback_parts.append("\n## Improvement Recommendations")
        feedback_parts.extend([f"- {rec}" for rec in recommendations])

        # Overall Assessment
        score_indicator = "basic" if metrics.word_count < 800 else "adequate"
        fact_status = (
            f", with {verified_count} verified and {unverified_count} unverified claims"
            if fact_check_results
            else ""
        )

        feedback_parts.append(f"\n## Overall Assessment")
        feedback_parts.append(
            f"This article provides {score_indicator} coverage of the topic{fact_status}. Focus on expanding content depth and ensuring factual accuracy."
        )

        return "\n".join(feedback_parts)

    def _parse_claims_from_response(
        self, response: str, article_content: str = ""
    ) -> List[str]:
        """Extract claims from XML or fallback to manual parsing."""
        claims = []

        # Method 1: Try XML parsing
        try:
            xml_match = re.search(
                r"<fact_check_analysis>.*?</fact_check_analysis>",
                response,
                re.DOTALL,
            )
            if xml_match:
                xml_content = xml_match.group()
                import xml.etree.ElementTree as ET

                root = ET.fromstring(xml_content)

                for claim_elem in root.findall("claim"):
                    text_elem = claim_elem.find("text")
                    if text_elem is not None and text_elem.text:
                        claim_text = text_elem.text.strip()
                        if claim_text and len(claim_text) > 10:
                            claims.append(claim_text)

                if claims:
                    logger.debug(f"Successfully parsed {len(claims)} claims from XML")
                    return claims[: self.max_claims_to_check]

        except (ET.ParseError, AttributeError) as e:
            logger.debug(f"XML parsing failed: {e}")

        # Method 2: Extract sentences from original article content that contain verifiable facts
        # This is more reliable than parsing LLM's potentially garbled response
        logger.debug("Using direct article parsing for claim extraction")

        # Look for sentences with quantifiable claims in the original content
        # We need to access the original article content - let's improve this method signature

        # For now, extract from the response but with better cleaning
        sentences = []
        for line in response.split("\n"):
            line = line.strip()
            if "." in line and len(line) > 20:
                # Split by sentences and clean
                sent_parts = line.split(".")
                for sent in sent_parts:
                    sent = sent.strip()
                    if len(sent) > 10:
                        sentences.append(sent)

        # Look for factual patterns in sentences
        fact_patterns = [
            r".*\d+%.*",  # Percentages
            r".*\$\d+.*(?:billion|million).*",  # Money amounts
            r".*\d+.*(?:patient|record|study|case).*",  # Medical statistics
            r".*(?:achieve|reach|show|process|detect).*\d+.*",  # Achievement claims
            r".*\d{4}.*(?:by|in|since).*",  # Date-based claims
        ]

        for sentence in sentences:
            for pattern in fact_patterns:
                if re.match(pattern, sentence, re.IGNORECASE):
                    # Clean up the sentence
                    claim = sentence.strip()
                    if not claim.endswith("."):
                        claim += "."

                    # Avoid duplicates and overly long claims
                    if (
                        len(claim) > 15
                        and len(claim) < 200
                        and claim not in claims
                        and not any(
                            existing in claim or claim in existing
                            for existing in claims
                        )
                    ):
                        claims.append(claim)
                        break

        # If still no claims and we have article content, extract directly from article
        if not claims and article_content:
            claims = self._extract_claims_directly_from_article(article_content)

        # Limit results and ensure quality
        result_claims = claims[: self.max_claims_to_check]
        logger.info(f"Extracted {len(result_claims)} claims using fallback parsing")
        return result_claims

    def _extract_claims_directly_from_article(self, article_content: str) -> List[str]:
        """
        Direct extraction of factual claims from article content.
        Used as fallback when LLM-based extraction fails.
        """
        claims = []

        # Split into sentences
        sentences = []
        for line in article_content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):  # Skip headings
                sent_parts = line.split(".")
                for sent in sent_parts:
                    sent = sent.strip()
                    if len(sent) > 20:  # Minimum length for claims
                        sentences.append(sent)

        # Pattern matching for factual claims
        fact_patterns = [
            (r".*\d+%.*(?:accuracy|detection|success|rate).*", "percentage accuracy"),
            (
                r".*\$\d+.*(?:billion|million|thousand).*(?:market|revenue|cost).*\d{4}.*",
                "financial projection",
            ),
            (
                r".*(?:over|more than|\d+).*(?:million|thousand).*(?:patient|record|case).*",
                "data volume",
            ),
            (r".*(?:achieve|reach|show|demonstrate).*\d+.*", "achievement claim"),
            (r".*\d{4}.*(?:by|in|since).*", "temporal claim"),
        ]

        for sentence in sentences:
            for pattern, claim_type in fact_patterns:
                if re.match(pattern, sentence, re.IGNORECASE):
                    claim = sentence.strip()
                    if not claim.endswith("."):
                        claim += "."

                    # Quality filters
                    if (
                        15 < len(claim) < 150
                        and claim not in claims
                        and not any(
                            existing in claim or claim in existing
                            for existing in claims
                        )
                    ):
                        claims.append(claim)
                        logger.debug(
                            f"Direct extraction found {claim_type}: {claim[:50]}..."
                        )
                        break

        return claims[: self.max_claims_to_check]

    def _fact_check_claims(self, claims: List[str]) -> List[FactCheckResult]:
        """
        Fact-check claims using research-based verification.
        Uses the comprehensive verify_claims_with_research tool.
        """
        fact_check_results = []

        if not claims:
            logger.info("No claims to fact-check")
            return fact_check_results

        try:
            if self.verify_claims_tool:
                # Convert claims list to comma-separated string for tool
                claims_str = ", ".join(claims)

                # Use comprehensive fact-checking tool
                verification_result = self.verify_claims_tool.invoke(
                    {"claims": claims_str}
                )

                if verification_result.get("success"):
                    verifications = verification_result.get("verifications", [])

                    for verification in verifications:
                        # Convert verification result to FactCheckResult
                        claim = verification.get("claim", "")
                        chunks_found = verification.get("relevant_chunks_found", 0)
                        chunk_contents = verification.get("chunk_contents", [])
                        verified = verification.get("verified", False)
                        confidence = verification.get("verification_confidence", "none")

                        # Create search results from chunk contents for compatibility
                        search_results = []
                        for chunk in chunk_contents:
                            search_results.append(
                                {
                                    "content": chunk.get("content", ""),
                                    "source": chunk.get("source", ""),
                                    "url": chunk.get("url", ""),
                                    "chunk_id": chunk.get("chunk_id", ""),
                                }
                            )

                        fact_check_results.append(
                            FactCheckResult(
                                claim=claim,
                                sources_found=chunks_found,
                                search_successful=True,
                                verified=verified if confidence != "none" else None,
                                search_results=search_results,
                            )
                        )

                        logger.debug(
                            f"Verified claim: '{claim[:50]}...' → {chunks_found} chunks, confidence: {confidence}"
                        )
                else:
                    error_msg = verification_result.get(
                        "error", "Unknown verification error"
                    )
                    logger.warning(f"Verification failed: {error_msg}")

                    # Create failed results for all claims
                    for claim in claims:
                        fact_check_results.append(
                            FactCheckResult(
                                claim=claim,
                                sources_found=0,
                                search_successful=False,
                                error=error_msg,
                            )
                        )
            else:
                logger.warning("verify_claims_tool not available, using fallback")
                # Fallback: create basic results
                for claim in claims:
                    fact_check_results.append(
                        FactCheckResult(
                            claim=claim,
                            sources_found=0,
                            search_successful=False,
                            error="Verification tool not available",
                        )
                    )

        except Exception as e:
            logger.error(f"Fact-checking failed: {e}")
            # Create error results for all claims
            for claim in claims:
                fact_check_results.append(
                    FactCheckResult(
                        claim=claim,
                        sources_found=0,
                        search_successful=False,
                        error=str(e),
                    )
                )

        logger.info(
            f"Fact-checking completed: {len(fact_check_results)} claims processed"
        )
        return fact_check_results

    # Deprecated _generate_qualitative_feedback method removed - now part of _comprehensive_review()

    def _calculate_overall_score(
        self, metrics: ArticleMetrics, fact_check_results: List[FactCheckResult]
    ) -> float:
        """Calculate overall score based on metrics and fact-checking."""

        import math

        # Start with neutral score
        score = 0.0
        logger.debug(f"Starting score calculation with metrics: {metrics}")

        # Content adequacy (sigmoid curve, not hard thresholds)
        word_count = metrics.word_count
        length_component = 0.0
        if word_count > 0:
            # Sigmoid: approaches 1 as word count approaches target
            length_score = 2 / (1 + math.exp(-(word_count - 1200) / 300)) - 1
            length_component = max(0, length_score) * 0.4
            score += length_component
            logger.debug(
                f"Length scoring: {word_count} words -> raw_score={length_score:.3f}, "
                f"component={length_component:.3f}"
            )

        # Structure adequacy
        heading_count = metrics.heading_count
        structure_component = 0.0
        if heading_count > 0:
            # Diminishing returns: 1 heading = 0.3, 2 = 0.55, 3 = 0.7, 4+ = 0.8
            structure_score = 1 - math.exp(-heading_count / 2.5)
            structure_component = structure_score * 0.3
            score += structure_component
            logger.debug(
                f"Structure scoring: {heading_count} headings -> raw_score={structure_score:.3f}, "
                f"component={structure_component:.3f}"
            )

        # Fact verification accuracy (fixed logic)
        fact_component = 0.0
        if fact_check_results and len(fact_check_results) > 0:
            verified_count = sum(1 for r in fact_check_results if r.verified == True)
            false_count = sum(1 for r in fact_check_results if r.verified == False)
            total_claims = len(fact_check_results)

            # Accuracy with penalty for false claims
            accuracy_rate = verified_count / total_claims
            false_penalty = false_count / total_claims
            fact_score = max(
                0, accuracy_rate - false_penalty * 2
            )  # False claims hurt more
            fact_component = fact_score * 0.3
            score += fact_component
            logger.debug(
                f"Fact checking: {verified_count}/{total_claims} verified, "
                f"{false_count} false -> component={fact_component:.3f}"
            )
        else:
            # No claims found - neutral (not bonus)
            logger.debug("No fact checking results available")

        final_score = max(0.0, min(1.0, score))
        logger.debug(
            f"Final score calculation: length={length_component:.3f} + "
            f"structure={structure_component:.3f} + fact={fact_component:.3f} = {final_score:.3f}"
        )

        return final_score

    def _parse_qualitative_feedback(
        self, feedback_text: str
    ) -> tuple[List[str], List[str]]:
        """Parse structured feedback into issues and recommendations."""

        # Extract issues from multiple sections that indicate problems
        issues = []
        issues.extend(self._extract_section(feedback_text, "## Content Quality Issues"))
        issues.extend(self._extract_section(feedback_text, "## Structural Assessment"))

        # Extract recommendations
        recommendations = self._extract_section(
            feedback_text, "## Improvement Recommendations"
        )

        # Fallback to old format if new format not found
        if not issues and not recommendations:
            issues = self._extract_section(feedback_text, "MAIN ISSUES:")
            recommendations = self._extract_section(feedback_text, "RECOMMENDATIONS:")

        # Ensure we have at least some feedback
        if not issues:
            issues = ["No specific issues identified"]
        if not recommendations:
            recommendations = ["Continue with current approach"]

        logger.debug(
            f"Parsed feedback: {len(issues)} issues, {len(recommendations)} recommendations"
        )
        return issues, recommendations

    def _extract_section(self, text: str, section_header: str) -> List[str]:
        """Extract bulleted items from a section with improved parsing."""

        # Try exact match first, then case-insensitive
        patterns = [
            re.escape(section_header) + r"\s*(.*?)(?=\n##|\n[A-Z]+:|$)",
            re.escape(section_header.replace("##", "").strip())
            + r"\s*(.*?)(?=\n##|\n[A-Z]+:|$)",
        ]

        section_content = ""
        for pattern in patterns:
            section_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if section_match:
                section_content = section_match.group(1).strip()
                break

        if not section_content:
            return []

        # Extract bullet points with various formats
        items = []
        for line in section_content.split("\n"):
            line = line.strip()
            # Handle different bullet formats: -, •, *, numbers
            if re.match(r"^[-•*]|\d+\.", line):
                item = re.sub(r"^[-•*]\s*|\d+\.\s*", "", line).strip()
                if item and len(item) > 5:  # Filter out very short items
                    items.append(item)

        # If no bullet points found but we have content, use paragraphs
        if not items and section_content:
            paragraphs = [p.strip() for p in section_content.split("\n\n") if p.strip()]
            if paragraphs:
                items = paragraphs[:3]  # Limit to avoid overly long lists

        return items
