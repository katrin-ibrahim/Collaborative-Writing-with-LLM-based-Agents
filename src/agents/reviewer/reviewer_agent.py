# src/agents/reviewer/reviewer_agent.py
import operator

import logging
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from typing import Annotated, Any, Dict, List, TypedDict

from src.agents.base_agent import BaseAgent
from src.agents.reviewer.data_models import (
    FeedbackCategory,
    FeedbackItem,
    ReviewError,
    ReviewFeedback,
    Severity,
)
from src.agents.reviewer.reviewer_tools import (
    analyze_article_structure,
    extract_verifiable_claims,
    fact_check_claim,
    generate_structured_feedback,
)
from src.utils.data import Article

logger = logging.getLogger(__name__)


class ReviewerState(TypedDict):
    """State management for ReviewerAgent workflow."""

    messages: Annotated[List[BaseMessage], operator.add]
    article: Dict[str, Any]  # Article data as dictionary
    extracted_claims: List[Dict[str, Any]]
    fact_check_results: List[Dict[str, Any]]
    structure_analysis: Dict[str, Any]
    feedback: Dict[str, Any]
    metadata: Dict[str, Any]


class ReviewerAgent(BaseAgent):
    """
    LangGraph-based agent for comprehensive article review.

    Workflow: extract_claims -> fact_check -> analyze_structure -> generate_feedback
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Create search tool with retrieval config from CLI args
        retrieval_config = config.get("retrieval_config")
        from src.agents.tools.search_toolkit import create_search_tool

        self.search_tool = create_search_tool(retrieval_config)

        self.workflow = self._build_workflow()
        self.logger = logging.getLogger(self.__class__.__name__)

    def process(self, article: Article) -> ReviewFeedback:
        """
        Main processing method for reviewing an article.

        Args:
            article: Article object to review

        Returns:
            ReviewFeedback object with complete review results
        """
        try:
            # Initialize state
            initial_state = ReviewerState(
                messages=[HumanMessage(content=f"Review article: {article.title}")],
                article=article.to_dict(),
                extracted_claims=[],
                fact_check_results=[],
                structure_analysis={},
                feedback={},
                metadata={"article_title": article.title},
            )

            # Run workflow
            final_state = self.workflow.invoke(initial_state)

            # Convert result to ReviewFeedback object
            feedback_data = final_state["feedback"]

            # Reconstruct category scores
            category_scores = {}
            for cat_str, score in feedback_data["category_scores"].items():
                category_scores[FeedbackCategory(cat_str)] = score

            # Reconstruct feedback items with validation
            issues = []
            for issue_data in feedback_data["issues"]:
                # Validate and clean category
                category_str = issue_data["category"]
                if "|" in category_str:
                    category_str = category_str.split("|")[
                        0
                    ]  # Take first part if multiple

                # Map to valid categories
                category_mapping = {
                    "factual": FeedbackCategory.FACTUAL,
                    "structural": FeedbackCategory.STRUCTURAL,
                    "structure": FeedbackCategory.STRUCTURAL,
                    "clarity": FeedbackCategory.CLARITY,
                    "completeness": FeedbackCategory.COMPLETENESS,
                    "style": FeedbackCategory.STYLE,
                }

                category = category_mapping.get(
                    category_str.lower(), FeedbackCategory.COMPLETENESS
                )

                # Validate and clean severity
                severity_str = issue_data["severity"]
                severity_mapping = {
                    "critical": Severity.CRITICAL,
                    "major": Severity.MAJOR,
                    "minor": Severity.MINOR,
                    "suggestion": Severity.SUGGESTION,
                }

                severity = severity_mapping.get(severity_str.lower(), Severity.MINOR)

                issues.append(
                    FeedbackItem(
                        category=category,
                        severity=severity,
                        description=issue_data["description"],
                        location=issue_data.get("location"),
                        suggestion=issue_data.get("suggestion", ""),
                        context=issue_data.get("context"),
                    )
                )

            return ReviewFeedback(
                overall_score=feedback_data["overall_score"],
                category_scores=category_scores,
                issues=issues,
                recommendations=feedback_data["recommendations"],
                summary=feedback_data["summary"],
                metadata=feedback_data["metadata"],
            )

        except Exception as e:
            self.logger.error(f"Review process failed: {e}")
            raise ReviewError(f"Failed to review article: {e}")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the reviewer agent."""
        workflow = StateGraph(ReviewerState)

        # Add nodes
        workflow.add_node("extract_claims", self._extract_claims_node)
        workflow.add_node("fact_check", self._fact_check_node)
        workflow.add_node("analyze_structure", self._analyze_structure_node)
        workflow.add_node("generate_feedback", self._generate_feedback_node)

        # Define workflow edges
        workflow.set_entry_point("extract_claims")
        workflow.add_edge("extract_claims", "fact_check")
        workflow.add_edge("fact_check", "analyze_structure")
        workflow.add_edge("analyze_structure", "generate_feedback")
        workflow.add_edge("generate_feedback", END)

        return workflow.compile()

    def _extract_claims_node(self, state: ReviewerState) -> ReviewerState:
        """Extract verifiable claims from the article content with LLM-driven analysis."""
        try:
            article_content = state["article"]["content"]
            article_title = state["article"]["title"]
            max_claims = self.config.get("reviewer.max_claims_per_article", 10)
            min_confidence = self.config.get("reviewer.min_claim_confidence", 0.7)

            # Enhanced LLM-driven claim extraction
            extraction_prompt = f"""
            Analyze the following article and extract verifiable factual claims that can be fact-checked.

            Article Title: {article_title}
            Article Content: {article_content}

            Instructions:
            1. Identify statements that make specific factual claims (dates, numbers, names, events)
            2. Exclude opinions, subjective statements, and general knowledge
            3. Focus on claims that can be verified against external sources
            4. Rate each claim's verifiability confidence (0.0-1.0)
            5. Extract up to {max_claims} most important claims

            Return your analysis in this JSON format:
            {{
                "claims": [
                    {{
                        "text": "exact claim text",
                        "section": "section where found",
                        "claim_type": "factual|statistical|historical|biographical",
                        "confidence": 0.8,
                        "reasoning": "why this claim is verifiable"
                    }}
                ],
                "total_sentences": number,
                "verifiable_claims_found": number
            }}
            """

            # Call LLM for sophisticated claim extraction
            llm_response = self.api_client.call_api(extraction_prompt)

            # Parse LLM response and fallback to simple extraction if needed
            try:
                import json
                import re

                json_match = re.search(r"\{.*\}", llm_response, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    extracted_claims = []

                    for claim_data in llm_result.get("claims", []):
                        if claim_data.get("confidence", 0) >= min_confidence:
                            extracted_claims.append(
                                {
                                    "text": claim_data["text"],
                                    "section": claim_data.get("section", "main"),
                                    "claim_type": claim_data.get(
                                        "claim_type", "factual"
                                    ),
                                    "confidence": claim_data.get("confidence", 0.8),
                                    "reasoning": claim_data.get("reasoning", ""),
                                }
                            )

                    state["extracted_claims"] = extracted_claims[:max_claims]

                else:
                    # Fallback to simple extraction
                    result = extract_verifiable_claims.invoke(
                        {"content": article_content, "max_claims": max_claims}
                    )
                    state["extracted_claims"] = result["claims"]

            except (json.JSONDecodeError, KeyError):
                # Fallback to simple extraction
                result = extract_verifiable_claims.invoke(
                    {"content": article_content, "max_claims": max_claims}
                )
                state["extracted_claims"] = result["claims"]

            # Filter by confidence threshold
            filtered_claims = [
                claim
                for claim in state["extracted_claims"]
                if claim.get("confidence", 0) >= min_confidence
            ]
            state["extracted_claims"] = filtered_claims

            # Handle edge cases
            if not state["extracted_claims"]:
                self.logger.warning("No verifiable claims found in article")
                state["messages"].append(
                    AIMessage(
                        content="No verifiable claims found that meet confidence threshold"
                    )
                )
            else:
                state["messages"].append(
                    AIMessage(
                        content=f"Extracted {len(state['extracted_claims'])} high-confidence claims for verification"
                    )
                )

            # Add metadata
            state["metadata"]["claims_extraction"] = {
                "total_extracted": len(state["extracted_claims"]),
                "confidence_threshold": min_confidence,
                "extraction_method": "llm_enhanced",
            }

            self.logger.info(
                f"Extracted {len(state['extracted_claims'])} claims with confidence >= {min_confidence}"
            )
            return state

        except Exception as e:
            self.logger.error(f"Claim extraction failed: {e}")
            state["extracted_claims"] = []
            state["messages"].append(AIMessage(content=f"Claim extraction failed: {e}"))
            state["metadata"]["claims_extraction"] = {
                "error": str(e),
                "extraction_method": "failed",
            }
            return state

    def _fact_check_node(self, state: ReviewerState) -> ReviewerState:
        """Fact-check the extracted claims with parallel processing and enhanced analysis."""
        try:
            claims = state["extracted_claims"]
            article_title = state["article"]["title"]
            self.config.get("reviewer.fact_check_timeout", 30)

            if not claims:
                self.logger.info("No claims to fact-check")
                state["fact_check_results"] = []
                state["messages"].append(
                    AIMessage(content="No claims available for fact-checking")
                )
                return state

            fact_check_results = []
            successful_checks = 0
            failed_checks = 0

            # Process each claim with enhanced fact-checking
            for i, claim_data in enumerate(claims):
                claim_text = claim_data["text"]
                claim_type = claim_data.get("claim_type", "factual")

                try:
                    # Enhanced fact-checking with LLM analysis
                    search_context = f"{article_title} {claim_type}"

                    # Get search results using the configured search tool
                    search_result = self.search_tool.invoke(
                        {"query": claim_text, "wiki_results": 3, "web_results": 2}
                    )

                    # Enhanced evidence analysis using LLM
                    if search_result["results"]:
                        evidence_analysis_prompt = f"""
                        Analyze the following claim against the provided evidence sources:

                        CLAIM TO VERIFY: {claim_text}
                        CLAIM TYPE: {claim_type}

                        EVIDENCE SOURCES:
                        {chr(10).join([f"Source {i+1}: {r['content'][:300]}..." for i, r in enumerate(search_result["results"][:3])])}

                        Provide your analysis in JSON format:
                        {{
                            "accuracy_score": 0.0-1.0,
                            "verification_status": "verified|disputed|unverifiable|needs_more_evidence",
                            "supporting_evidence": ["evidence1", "evidence2"],
                            "contradicting_evidence": ["contradiction1"],
                            "confidence": 0.0-1.0,
                            "reasoning": "detailed explanation"
                        }}
                        """

                        llm_analysis = self.api_client.call_api(
                            evidence_analysis_prompt
                        )

                        # Parse LLM analysis with fallback
                        try:
                            import json
                            import re

                            json_match = re.search(r"\{.*\}", llm_analysis, re.DOTALL)
                            if json_match:
                                analysis = json.loads(json_match.group())

                                result = {
                                    "claim": claim_text,
                                    "accuracy_score": analysis.get(
                                        "accuracy_score", 0.5
                                    ),
                                    "supporting_evidence": analysis.get(
                                        "supporting_evidence", []
                                    ),
                                    "contradicting_evidence": analysis.get(
                                        "contradicting_evidence", []
                                    ),
                                    "verification_status": analysis.get(
                                        "verification_status", "unverifiable"
                                    ),
                                    "sources": [
                                        r["source"] for r in search_result["results"]
                                    ],
                                    "evidence_count": len(search_result["results"]),
                                    "confidence": analysis.get("confidence", 0.5),
                                    "reasoning": analysis.get("reasoning", ""),
                                    "claim_type": claim_type,
                                }
                            else:
                                # Fallback to simple fact-checking
                                result = fact_check_claim.invoke(
                                    {
                                        "claim_text": claim_text,
                                        "context": search_context,
                                    }
                                )
                                result["claim_type"] = claim_type

                        except (json.JSONDecodeError, KeyError):
                            # Fallback to simple fact-checking
                            result = fact_check_claim.invoke(
                                {"claim_text": claim_text, "context": search_context}
                            )
                            result["claim_type"] = claim_type
                    else:
                        # No evidence found
                        result = {
                            "claim": claim_text,
                            "accuracy_score": 0.0,
                            "supporting_evidence": [],
                            "contradicting_evidence": [],
                            "verification_status": "unverifiable",
                            "sources": [],
                            "evidence_count": 0,
                            "confidence": 0.0,
                            "reasoning": "No evidence sources found",
                            "claim_type": claim_type,
                        }

                    fact_check_results.append(result)
                    successful_checks += 1

                    # Log progress for long-running fact-checks
                    if len(claims) > 3:
                        self.logger.info(
                            f"Fact-checked claim {i+1}/{len(claims)}: {result['verification_status']}"
                        )

                except Exception as claim_error:
                    self.logger.error(
                        f"Failed to fact-check claim '{claim_text[:50]}...': {claim_error}"
                    )

                    # Add failed result with error info
                    result = {
                        "claim": claim_text,
                        "accuracy_score": 0.0,
                        "supporting_evidence": [],
                        "contradicting_evidence": [],
                        "verification_status": "error",
                        "sources": [],
                        "evidence_count": 0,
                        "confidence": 0.0,
                        "reasoning": f"Fact-check failed: {str(claim_error)}",
                        "claim_type": claim_type,
                        "error": str(claim_error),
                    }
                    fact_check_results.append(result)
                    failed_checks += 1

            state["fact_check_results"] = fact_check_results

            # Calculate summary statistics
            verified_count = sum(
                1 for r in fact_check_results if r["verification_status"] == "verified"
            )
            disputed_count = sum(
                1 for r in fact_check_results if r["verification_status"] == "disputed"
            )
            unverifiable_count = sum(
                1
                for r in fact_check_results
                if r["verification_status"] == "unverifiable"
            )

            # Add metadata
            state["metadata"]["fact_checking"] = {
                "total_claims": len(claims),
                "successful_checks": successful_checks,
                "failed_checks": failed_checks,
                "verified": verified_count,
                "disputed": disputed_count,
                "unverifiable": unverifiable_count,
                "average_accuracy": sum(r["accuracy_score"] for r in fact_check_results)
                / max(1, len(fact_check_results)),
            }

            state["messages"].append(
                AIMessage(
                    content=f"Fact-checked {len(fact_check_results)} claims: {verified_count} verified, {disputed_count} disputed, {unverifiable_count} unverifiable"
                )
            )

            self.logger.info(
                f"Fact-checking completed: {verified_count}/{len(fact_check_results)} claims verified"
            )
            return state

        except Exception as e:
            self.logger.error(f"Fact-checking failed: {e}")
            state["fact_check_results"] = []
            state["messages"].append(AIMessage(content=f"Fact-checking failed: {e}"))
            state["metadata"]["fact_checking"] = {
                "error": str(e),
                "total_claims": len(state.get("extracted_claims", [])),
                "successful_checks": 0,
            }
            return state

    def _analyze_structure_node(self, state: ReviewerState) -> ReviewerState:
        """Analyze the article's structure and organization with detailed evaluation."""
        try:
            article = state["article"]
            title = article.get("title", "")
            content = article.get("content", "")
            sections = article.get("sections", {})
            article.get("outline", {})

            # Enhanced structure analysis using LLM
            structure_analysis_prompt = f"""
            Analyze the structure and organization of this article:

            Title: {title}
            Content Length: {len(content)} characters
            Number of Sections: {len(sections)}

            Article Content:
            {content}

            Section Breakdown:
            {chr(10).join([f"- {k}: {v[:100]}..." for k, v in sections.items()]) if sections else "No explicit sections"}

            Evaluate the following aspects (score 0.0-1.0):
            1. Outline Logic: Does the article follow a logical progression?
            2. Section Transitions: Are transitions between sections smooth?
            3. Content Coherence: Is the content well-organized within sections?
            4. Topic Completeness: Does the article adequately cover the topic?
            5. Paragraph Organization: Are paragraphs well-structured?

            Identify specific issues and provide actionable recommendations.

            Return analysis in JSON format:
            {{
                "outline_score": 0.8,
                "transition_score": 0.7,
                "coherence_score": 0.9,
                "completeness_score": 0.6,
                "paragraph_organization_score": 0.8,
                "issues": [
                    {{
                        "type": "transition|coherence|completeness|organization",
                        "description": "specific issue description",
                        "location": "section or paragraph reference",
                        "severity": "minor|major|critical"
                    }}
                ],
                "recommendations": [
                    {{
                        "action": "specific recommendation",
                        "priority": "high|medium|low",
                        "rationale": "why this improvement is needed"
                    }}
                ],
                "strengths": ["what works well in the structure"],
                "overall_assessment": "summary of structural quality"
            }}
            """

            llm_analysis = self.api_client.call_api(structure_analysis_prompt)

            # Parse LLM analysis with enhanced fallback
            try:
                import json
                import re

                json_match = re.search(r"\{.*\}", llm_analysis, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())

                    # Calculate overall structure score
                    scores = [
                        analysis.get("outline_score", 0.5),
                        analysis.get("transition_score", 0.5),
                        analysis.get("coherence_score", 0.5),
                        analysis.get("completeness_score", 0.5),
                        analysis.get("paragraph_organization_score", 0.5),
                    ]
                    overall_score = sum(scores) / len(scores)

                    # Process issues and recommendations
                    issues = []
                    recommendations = []

                    for issue in analysis.get("issues", []):
                        issues.append(
                            f"[{issue.get('severity', 'minor').upper()}] {issue.get('description', 'Structural issue')}"
                        )

                    for rec in analysis.get("recommendations", []):
                        priority = rec.get("priority", "medium").upper()
                        recommendations.append(
                            f"[{priority}] {rec.get('action', 'Improve structure')}"
                        )

                    result = {
                        "outline_score": analysis.get("outline_score", 0.5),
                        "transition_score": analysis.get("transition_score", 0.5),
                        "coherence_score": analysis.get("coherence_score", 0.5),
                        "completeness_score": analysis.get("completeness_score", 0.5),
                        "paragraph_organization_score": analysis.get(
                            "paragraph_organization_score", 0.5
                        ),
                        "overall_structure_score": overall_score,
                        "issues": issues,
                        "recommendations": recommendations,
                        "strengths": analysis.get("strengths", []),
                        "overall_assessment": analysis.get(
                            "overall_assessment", "Structure analysis completed"
                        ),
                        "sections_count": len(sections),
                        "content_length": len(content),
                        "analysis_method": "llm_enhanced",
                    }
                else:
                    # Fallback to simple analysis
                    result = analyze_article_structure.invoke({"article_dict": article})
                    result["analysis_method"] = "simple_fallback"

            except (json.JSONDecodeError, KeyError) as parse_error:
                self.logger.warning(
                    f"Failed to parse LLM structure analysis: {parse_error}"
                )
                # Fallback to simple analysis
                result = analyze_article_structure.invoke({"article_dict": article})
                result["analysis_method"] = "simple_fallback"
                result["parse_error"] = str(parse_error)

            # Add detailed structural metrics
            result.update(
                {
                    "word_count": len(content.split()),
                    "sentence_count": len([s for s in content.split(".") if s.strip()]),
                    "paragraph_count": len(
                        [p for p in content.split("\n\n") if p.strip()]
                    ),
                    "has_title": bool(title),
                    "has_sections": len(sections) > 0,
                    "section_balance": (
                        self._calculate_section_balance(sections) if sections else 0.0
                    ),
                }
            )

            state["structure_analysis"] = result

            # Add metadata
            state["metadata"]["structure_analysis"] = {
                "method": result.get("analysis_method", "unknown"),
                "overall_score": result["overall_structure_score"],
                "issues_found": len(result["issues"]),
                "recommendations_made": len(result["recommendations"]),
            }

            state["messages"].append(
                AIMessage(
                    content=f"Structure analysis completed: score {result['overall_structure_score']:.2f}, {len(result['issues'])} issues found"
                )
            )

            self.logger.info(
                f"Structure analysis completed with score {result['overall_structure_score']:.2f}"
            )
            return state

        except Exception as e:
            self.logger.error(f"Structure analysis failed: {e}")
            # Enhanced fallback structure analysis
            content_length = len(state["article"].get("content", ""))
            sections_count = len(state["article"].get("sections", {}))

            fallback_score = min(
                0.8,
                max(0.3, (content_length / 1000) * 0.5 + (sections_count / 5) * 0.3),
            )

            state["structure_analysis"] = {
                "outline_score": fallback_score,
                "transition_score": fallback_score,
                "coherence_score": fallback_score,
                "completeness_score": fallback_score,
                "paragraph_organization_score": fallback_score,
                "overall_structure_score": fallback_score,
                "issues": [f"Structure analysis failed: {e}"],
                "recommendations": [
                    "Manual structural review recommended due to analysis failure"
                ],
                "strengths": [],
                "overall_assessment": f"Analysis failed, using fallback score: {fallback_score:.2f}",
                "sections_count": sections_count,
                "content_length": content_length,
                "analysis_method": "error_fallback",
                "error": str(e),
            }

            state["metadata"]["structure_analysis"] = {
                "method": "error_fallback",
                "error": str(e),
                "fallback_score": fallback_score,
            }

            state["messages"].append(
                AIMessage(
                    content=f"Structure analysis failed, using fallback assessment: {e}"
                )
            )
            return state

    def _calculate_section_balance(self, sections: Dict[str, str]) -> float:
        """Calculate how balanced the sections are in terms of content length."""
        if not sections or len(sections) < 2:
            return 1.0

        lengths = [len(content) for content in sections.values()]
        if not lengths:
            return 0.0

        avg_length = sum(lengths) / len(lengths)
        if avg_length == 0:
            return 0.0

        # Calculate coefficient of variation (lower is more balanced)
        variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
        std_dev = variance**0.5
        cv = std_dev / avg_length

        # Convert to balance score (0.0 = unbalanced, 1.0 = perfectly balanced)
        balance_score = max(0.0, 1.0 - min(1.0, cv))
        return balance_score

    def _generate_feedback_node(self, state: ReviewerState) -> ReviewerState:
        """Generate comprehensive structured feedback with prioritization and actionable suggestions."""
        try:
            fact_check_results = state.get("fact_check_results", [])
            structure_analysis = state.get("structure_analysis", {})
            article = state["article"]

            # Enhanced feedback generation using LLM
            feedback_prompt = f"""
            Generate comprehensive review feedback for this article based on the analysis results:

            ARTICLE: {article.get('title', 'Untitled')}
            Content Length: {len(article.get('content', ''))} characters

            FACT-CHECKING RESULTS:
            Total Claims Analyzed: {len(fact_check_results)}
            {chr(10).join([f"- {r['claim'][:100]}... | Status: {r['verification_status']} | Score: {r['accuracy_score']:.2f}" for r in fact_check_results[:5]])}

            STRUCTURE ANALYSIS:
            Overall Score: {structure_analysis.get('overall_structure_score', 0.5):.2f}
            Issues: {structure_analysis.get('issues', [])}
            Recommendations: {structure_analysis.get('recommendations', [])}

            Generate detailed feedback with:
            1. Category-specific scores (factual, structural, clarity, completeness, style)
            2. Prioritized issues with severity levels
            3. Actionable recommendations
            4. Overall assessment and summary

            Return in JSON format:
            {{
                "category_scores": {{
                    "factual": 0.8,
                    "structural": 0.7,
                    "clarity": 0.8,
                    "completeness": 0.6,
                    "style": 0.7
                }},
                "issues": [
                    {{
                        "category": "factual|structural|clarity|completeness|style",
                        "severity": "critical|major|minor|suggestion",
                        "description": "specific issue description",
                        "location": "section or line reference",
                        "suggestion": "actionable recommendation",
                        "context": "additional context or explanation"
                    }}
                ],
                "recommendations": [
                    {{
                        "priority": "high|medium|low",
                        "action": "specific action to take",
                        "category": "factual|structural|clarity|completeness|style",
                        "impact": "expected improvement from this action"
                    }}
                ],
                "strengths": ["what the article does well"],
                "summary": "overall assessment of the article quality",
                "next_steps": ["immediate actions for improvement"]
            }}
            """

            llm_feedback = self.api_client.call_api(feedback_prompt)

            # Parse LLM feedback with comprehensive fallback
            try:
                import json
                import re

                json_match = re.search(r"\{.*\}", llm_feedback, re.DOTALL)
                if json_match:
                    feedback_data = json.loads(json_match.group())

                    # Process and validate category scores
                    category_scores = feedback_data.get("category_scores", {})

                    # Ensure all required categories are present
                    required_categories = [
                        "factual",
                        "structural",
                        "clarity",
                        "completeness",
                        "style",
                    ]
                    for category in required_categories:
                        if category not in category_scores:
                            if category == "factual":
                                # Calculate factual score from fact-check results
                                if fact_check_results:
                                    verified_ratio = sum(
                                        1
                                        for r in fact_check_results
                                        if r["verification_status"] == "verified"
                                    ) / len(fact_check_results)
                                    avg_accuracy = sum(
                                        r["accuracy_score"] for r in fact_check_results
                                    ) / len(fact_check_results)
                                    category_scores["factual"] = (
                                        verified_ratio * 0.6
                                    ) + (avg_accuracy * 0.4)
                                else:
                                    category_scores["factual"] = (
                                        0.8  # Default if no claims
                                    )
                            elif category == "structural":
                                category_scores["structural"] = structure_analysis.get(
                                    "overall_structure_score", 0.5
                                )
                            else:
                                category_scores[category] = 0.7  # Default score

                    # Process issues with enhanced categorization
                    processed_issues = []
                    for issue in feedback_data.get("issues", []):
                        processed_issues.append(
                            {
                                "category": issue.get("category", "completeness"),
                                "severity": issue.get("severity", "minor"),
                                "description": issue.get(
                                    "description", "Issue identified"
                                ),
                                "location": issue.get("location"),
                                "suggestion": issue.get(
                                    "suggestion", "Review and improve"
                                ),
                                "context": issue.get("context"),
                            }
                        )

                    # Add fact-checking issues
                    for result in fact_check_results:
                        if (
                            result["verification_status"]
                            in ["disputed", "unverifiable"]
                            or result["accuracy_score"] < 0.5
                        ):
                            severity = (
                                "critical"
                                if result["accuracy_score"] < 0.3
                                else (
                                    "major"
                                    if result["accuracy_score"] < 0.6
                                    else "minor"
                                )
                            )
                            processed_issues.append(
                                {
                                    "category": "factual",
                                    "severity": severity,
                                    "description": f"Claim verification issue: {result['claim'][:100]}...",
                                    "location": "content",
                                    "suggestion": f"Verify claim with additional sources or remove if unsubstantiated",
                                    "context": f"Status: {result['verification_status']}, Score: {result['accuracy_score']:.2f}",
                                }
                            )

                    # Add structural issues
                    for issue in structure_analysis.get("issues", []):
                        processed_issues.append(
                            {
                                "category": "structural",
                                "severity": "minor",
                                "description": issue,
                                "location": "structure",
                                "suggestion": "Improve article organization",
                                "context": "Structural analysis",
                            }
                        )

                    # Process recommendations
                    recommendations = []
                    for rec in feedback_data.get("recommendations", []):
                        recommendations.append(
                            f"[{rec.get('priority', 'medium').upper()}] {rec.get('action', 'Improve content')}"
                        )

                    # Add structure recommendations
                    recommendations.extend(
                        structure_analysis.get("recommendations", [])
                    )

                    # Calculate overall score
                    overall_score = sum(category_scores.values()) / len(category_scores)

                    # Generate comprehensive summary
                    verified_claims = sum(
                        1
                        for r in fact_check_results
                        if r["verification_status"] == "verified"
                    )
                    total_claims = len(fact_check_results)

                    summary_parts = [
                        f"Overall quality score: {overall_score:.2f}",
                        (
                            f"Fact-checking: {verified_claims}/{total_claims} claims verified"
                            if total_claims > 0
                            else "No verifiable claims found"
                        ),
                        f"Structure score: {structure_analysis.get('overall_structure_score', 0.5):.2f}",
                        f"{len(processed_issues)} issues identified",
                        feedback_data.get("summary", "Review completed"),
                    ]

                    result = {
                        "overall_score": overall_score,
                        "category_scores": category_scores,
                        "issues": processed_issues,
                        "recommendations": recommendations[
                            :10
                        ],  # Limit recommendations
                        "summary": " | ".join(summary_parts),
                        "strengths": feedback_data.get("strengths", []),
                        "next_steps": feedback_data.get("next_steps", []),
                        "metadata": {
                            "claims_analyzed": total_claims,
                            "claims_verified": verified_claims,
                            "structure_score": structure_analysis.get(
                                "overall_structure_score", 0.5
                            ),
                            "issues_found": len(processed_issues),
                            "feedback_method": "llm_enhanced",
                            "review_timestamp": "2025-01-08",  # Would use actual timestamp
                        },
                    }
                else:
                    # Fallback to simple feedback generation
                    result = generate_structured_feedback.invoke(
                        {
                            "claims_results": fact_check_results,
                            "structure_results": structure_analysis,
                            "article_dict": article,
                        }
                    )
                    result["metadata"]["feedback_method"] = "simple_fallback"

            except (json.JSONDecodeError, KeyError) as parse_error:
                self.logger.warning(f"Failed to parse LLM feedback: {parse_error}")
                # Fallback to simple feedback generation
                result = generate_structured_feedback.invoke(
                    {
                        "claims_results": fact_check_results,
                        "structure_results": structure_analysis,
                        "article_dict": article,
                    }
                )
                result["metadata"]["feedback_method"] = "simple_fallback"
                result["metadata"]["parse_error"] = str(parse_error)

            # Final validation and cleanup
            result = self._validate_and_clean_feedback(result)

            state["feedback"] = result
            state["messages"].append(
                AIMessage(
                    content=f"Comprehensive review completed: {result['summary']}"
                )
            )

            self.logger.info(
                f"Feedback generation completed: {len(result['issues'])} issues, score {result['overall_score']:.2f}"
            )
            return state

        except Exception as e:
            self.logger.error(f"Feedback generation failed: {e}")

            # Enhanced fallback feedback
            fact_check_results = state.get("fact_check_results", [])
            structure_analysis = state.get("structure_analysis", {})

            # Calculate basic scores from available data
            factual_score = 0.8
            if fact_check_results:
                verified_count = sum(
                    1
                    for r in fact_check_results
                    if r.get("verification_status") == "verified"
                )
                factual_score = (
                    verified_count / len(fact_check_results)
                    if fact_check_results
                    else 0.8
                )

            structural_score = structure_analysis.get("overall_structure_score", 0.5)
            overall_score = (factual_score + structural_score) / 2

            state["feedback"] = {
                "overall_score": overall_score,
                "category_scores": {
                    "factual": factual_score,
                    "structural": structural_score,
                    "clarity": 0.6,
                    "completeness": 0.6,
                    "style": 0.6,
                },
                "issues": [
                    {
                        "category": "completeness",
                        "severity": "major",
                        "description": f"Review process encountered an error: {str(e)}",
                        "location": "system",
                        "suggestion": "Manual review recommended",
                        "context": "Automated review failed",
                    }
                ],
                "recommendations": [
                    "Manual review recommended due to processing error",
                    "Verify all factual claims independently",
                    "Review article structure and organization",
                ],
                "summary": f"Partial review completed with errors (score: {overall_score:.2f}). Manual review recommended.",
                "metadata": {
                    "error": str(e),
                    "feedback_method": "error_fallback",
                    "partial_results": True,
                },
            }

            state["messages"].append(
                AIMessage(
                    content=f"Feedback generation failed, providing fallback assessment: {e}"
                )
            )
            return state

    def _validate_and_clean_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the feedback data structure."""
        # Ensure all required fields are present
        required_fields = [
            "overall_score",
            "category_scores",
            "issues",
            "recommendations",
            "summary",
            "metadata",
        ]
        for field in required_fields:
            if field not in feedback:
                if field == "overall_score":
                    feedback[field] = 0.5
                elif field == "category_scores":
                    feedback[field] = {
                        "factual": 0.5,
                        "structural": 0.5,
                        "clarity": 0.5,
                        "completeness": 0.5,
                        "style": 0.5,
                    }
                elif field in ["issues", "recommendations"]:
                    feedback[field] = []
                elif field == "summary":
                    feedback[field] = "Review completed"
                elif field == "metadata":
                    feedback[field] = {}

        # Validate score ranges
        feedback["overall_score"] = max(0.0, min(1.0, feedback["overall_score"]))
        for category in feedback["category_scores"]:
            feedback["category_scores"][category] = max(
                0.0, min(1.0, feedback["category_scores"][category])
            )

        # Limit number of issues and recommendations
        feedback["issues"] = feedback["issues"][:20]  # Max 20 issues
        feedback["recommendations"] = feedback["recommendations"][
            :15
        ]  # Max 15 recommendations

        return feedback
