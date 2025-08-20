# src/collaborative/agents/writer_agent.py
"""
Refactored WriterAgent using clean architecture with real tools only.
"""

import logging
import re
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from typing import Any, Dict, List, Optional

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    improvement_prompt,
    knowledge_organization_prompt,
    planning_prompt,
    refinement_prompt,
    section_content_prompt_with_research,
    section_content_prompt_without_research,
)
from src.collaborative.data_models import Outline, ReviewFeedback, WriterState
from src.collaborative.tools.writer_toolkit import WriterToolkit
from src.utils.data import Article
from src.utils.io import create_error_article

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """
    Sophisticated Writer agent with planning-first workflow.

    Workflow: plan_outline → targeted_research → refine_outline → write_content
    Uses only real tools (search_and_retrieve) and LLM reasoning.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get configuration values with proper defaults
        self.max_research_iterations = config.get("writer.max_research_iterations", 3)
        self.knowledge_coverage_threshold = config.get(
            "writer.knowledge_coverage_threshold", 0.7
        )
        self.max_research_queries_per_iteration = config.get(
            "writer.max_queries_per_iteration", 6
        )
        self.max_search_results = config.get("writer.max_search_results", 5)
        self.rm_type = config.get("retrieval_manager_type", "wiki")

        # Initialize writer toolkit (only search_and_retrieve tool)
        self.toolkit = WriterToolkit(config)

        # Get the search tool
        self.search_tool = None
        for tool in self.toolkit.get_available_tools():
            if tool.name == "search_and_retrieve":
                self.search_tool = tool
                break

        # Build workflow
        self.workflow = self._build_workflow()

        logger.info(
            f"WriterAgent initialized with planning-first workflow "
            f"(max_iterations={self.max_research_iterations}, "
            f"coverage_threshold={self.knowledge_coverage_threshold})"
        )

    def _build_workflow(self) -> StateGraph:
        """Build sophisticated planning-first workflow."""
        workflow = StateGraph(WriterState)

        # Add workflow nodes
        workflow.add_node("plan_outline", self._plan_outline_node)
        workflow.add_node("targeted_research", self._targeted_research_node)
        workflow.add_node("refine_outline", self._refine_outline_node)
        workflow.add_node("write_content", self._write_content_node)

        # Set entry point to planning
        workflow.set_entry_point("plan_outline")

        # Planning → Research (always)
        workflow.add_edge("plan_outline", "targeted_research")

        # Research → Refinement (always)
        workflow.add_edge("targeted_research", "refine_outline")

        # Refinement → Conditional (research more or write)
        workflow.add_conditional_edges(
            "refine_outline",
            self._decide_after_outline_refinement,
            {"research_more": "targeted_research", "ready_to_write": "write_content"},
        )

        # Writing → End
        workflow.add_edge("write_content", END)

        return workflow.compile()

    def process(
        self,
        topic: str,
        previous_article: Optional[Article] = None,
        review_feedback: Optional[ReviewFeedback] = None,
    ) -> Article:
        """Process topic through sophisticated planning-first workflow."""

        logger.info(f"Starting writer workflow for: {topic}")

        # Handle improvement case
        if previous_article and review_feedback:
            return self._improve_article(previous_article, review_feedback, topic)

        # Initialize state for new article
        initial_state = WriterState(
            messages=[HumanMessage(content=f"Write an article about: {topic}")],
            topic=topic,
            initial_outline=None,
            research_queries=[],
            search_results=[],
            organized_knowledge=None,
            refined_outline=None,
            knowledge_gaps=[],
            article_content="",
            research_iterations=0,
            ready_to_write=False,
            metadata={
                "method": "writer_agent",
                "workflow_version": "3.0",
                "tool_usage": [],
                "decisions": [],
            },
        )

        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            article = self._create_article_from_state(final_state)

            logger.info(
                f"Completed writer workflow for '{topic}' "
                f"({final_state.research_iterations} research iterations, "
                f"{len(final_state.search_results)} sources)"
            )

            return article

        except Exception as e:
            logger.error(f"Writer workflow failed for '{topic}': {e}")
            return create_error_article(topic, str(e))

    def _improve_article(
        self, article: Article, feedback: ReviewFeedback, topic: str
    ) -> Article:
        """Improve article based on reviewer feedback."""

        logger.info(
            f"Improving article based on feedback: score {feedback.overall_score:.3f}"
        )

        try:
            # Create improvement prompt
            prompt = improvement_prompt(
                topic=topic,
                article_content=article.content,
                feedback_text=feedback.feedback_text,
                recommendations="\n".join(
                    [f"- {rec}" for rec in feedback.recommendations]
                ),
            )

            # Generate improved content
            improved_content = self.api_client.call_api(prompt)

            if not improved_content or len(improved_content.strip()) < 100:
                logger.warning("Generated improvement is too short, returning original")
                return article

            # Create improved article
            improved_article = Article(
                title=article.title,
                content=improved_content,
                sections=article.sections,
                metadata={
                    **article.metadata,
                    "improved": True,
                    "original_score": feedback.overall_score,
                    "improvement_feedback": (
                        feedback.feedback_text[:200] + "..."
                        if len(feedback.feedback_text) > 200
                        else feedback.feedback_text
                    ),
                },
            )

            logger.info(f"Article improvement completed for: {topic}")
            return improved_article

        except Exception as e:
            logger.error(f"Article improvement failed: {e}")
            return article  # Return original on failure

    def _plan_outline_node(self, state: WriterState) -> WriterState:
        """Create initial outline using LLM reasoning."""

        logger.info(f"Planning initial outline for: {state.topic}")

        # Use template prompt
        prompt = planning_prompt(state.topic)
        outline_response = self.api_client.call_api(prompt)
        initial_outline = self._parse_outline_response(outline_response, state.topic)

        # Generate targeted research queries based on outline
        research_queries = self._generate_research_queries_from_outline(
            initial_outline, state.topic
        )

        state.initial_outline = initial_outline
        state.research_queries = research_queries
        state.metadata["decisions"].append(
            f"Created initial outline with {len(initial_outline.headings)} sections"
        )
        state.metadata["decisions"].append(
            f"Generated {len(research_queries)} targeted research queries"
        )

        logger.info(
            f"Created outline with {len(initial_outline.headings)} sections, "
            f"generated {len(research_queries)} research queries"
        )

        return state

    def _targeted_research_node(self, state: WriterState) -> WriterState:
        """Conduct targeted research using search tool."""

        state.research_iterations += 1
        logger.info(
            f"Starting research iteration {state.research_iterations} "
            f"with {len(state.research_queries)} queries"
        )

        # Limit queries per iteration for efficiency
        current_queries = state.research_queries[
            : self.max_research_queries_per_iteration
        ]

        new_results = []
        for query in current_queries:
            try:
                # Use real search tool
                search_result = self.search_tool.invoke(
                    {
                        "query": query,
                        "rm_type": self.rm_type,
                        "max_results": self.max_search_results,
                        "purpose": "writing",
                    }
                )

                if search_result.get("success") and search_result.get("results"):
                    new_results.extend(search_result["results"])
                    state.metadata["tool_usage"].append(
                        f"search_and_retrieve: '{query}' → {len(search_result['results'])} results"
                    )

            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # Add new results to existing ones
        state.search_results.extend(new_results)

        # Organize knowledge using LLM
        if state.search_results:
            try:
                organized = self._organize_search_results_with_llm(
                    state.topic, state.search_results
                )
                state.organized_knowledge = organized
                state.metadata["tool_usage"].append(
                    f"organize_knowledge_llm: {len(state.search_results)} results organized"
                )

            except Exception as e:
                logger.warning(f"Knowledge organization failed: {e}")

        logger.info(
            f"Research iteration {state.research_iterations} completed: "
            f"{len(new_results)} new results, {len(state.search_results)} total"
        )

        return state

    def _refine_outline_node(self, state: WriterState) -> WriterState:
        """Refine outline based on research findings using LLM reasoning."""

        logger.info("Refining outline based on research findings")

        if not state.organized_knowledge:
            # No knowledge to work with, keep original outline
            state.refined_outline = state.initial_outline
            state.ready_to_write = True
            return state

        # Analyze knowledge coverage for each outline section
        knowledge_coverage = self._analyze_knowledge_coverage(
            state.initial_outline, state.organized_knowledge
        )

        # Use template prompt for refinement
        prompt = refinement_prompt(
            topic=state.topic,
            current_outline=self._format_outline_for_prompt(state.initial_outline),
            knowledge_summary=state.organized_knowledge.get(
                "summary", "No summary available"
            ),
            coverage_analysis=str(knowledge_coverage),
        )

        refined_response = self.api_client.call_api(prompt)
        refined_outline = self._parse_outline_response(refined_response, state.topic)

        # Identify knowledge gaps for potential additional research
        knowledge_gaps = self._identify_knowledge_gaps(
            refined_outline, state.organized_knowledge
        )

        state.refined_outline = refined_outline
        state.knowledge_gaps = knowledge_gaps

        state.metadata["decisions"].append(
            f"Refined outline: {len(state.initial_outline.headings)} → "
            f"{len(refined_outline.headings)} sections"
        )

        if knowledge_gaps:
            state.metadata["decisions"].append(
                f"Identified {len(knowledge_gaps)} knowledge gaps"
            )

        logger.info(
            f"Outline refined: {len(state.initial_outline.headings)} → "
            f"{len(refined_outline.headings)} sections, "
            f"{len(knowledge_gaps)} gaps identified"
        )

        return state

    def _write_content_node(self, state: WriterState) -> WriterState:
        """Generate article content using LLM with organized knowledge."""

        logger.info("Generating article content")

        working_outline = state.refined_outline or state.initial_outline

        if not working_outline:
            state.article_content = (
                f"# {state.topic}\n\nUnable to create outline for article."
            )
            return state

        # Initialize article with title
        article_parts = [f"# {working_outline.title}"]

        # Generate each section using organized knowledge
        for section_heading in working_outline.headings:
            try:
                section_content = self._generate_section_content(
                    section_heading, state.topic, state.organized_knowledge
                )

                article_parts.append(f"## {section_heading}")
                article_parts.append(section_content)

            except Exception as e:
                logger.warning(f"Failed to generate section '{section_heading}': {e}")
                article_parts.append(f"## {section_heading}")
                article_parts.append("Content generation failed for this section.")

        # Combine all parts
        full_content = "\n\n".join(article_parts)
        state.article_content = full_content

        state.metadata["decisions"].append(
            f"Generated article with {len(working_outline.headings)} sections"
        )

        logger.info(
            f"Article generation completed: {len(full_content)} characters, "
            f"{len(working_outline.headings)} sections"
        )

        return state

    def _decide_after_outline_refinement(self, state: WriterState) -> str:
        """Decide whether to research more or proceed to writing."""

        # Check iteration limit
        if state.research_iterations >= self.max_research_iterations:
            logger.info("Max research iterations reached, proceeding to writing")
            state.ready_to_write = True
            return "ready_to_write"

        # Check knowledge coverage
        if state.organized_knowledge:
            coverage_score = state.organized_knowledge.get("coverage_score", 0)

            if coverage_score >= self.knowledge_coverage_threshold:
                logger.info(
                    f"Knowledge coverage sufficient ({coverage_score:.2f}), proceeding to writing"
                )
                state.ready_to_write = True
                return "ready_to_write"

        # Check if we have significant knowledge gaps
        if len(state.knowledge_gaps) > 2:
            # Generate new research queries for gaps
            gap_queries = [
                f"detailed information about {gap} in context of {state.topic}"
                for gap in state.knowledge_gaps[:3]  # Limit new queries
            ]

            state.research_queries = gap_queries
            logger.info(
                f"Knowledge gaps found, conducting additional research for {len(gap_queries)} topics"
            )
            return "research_more"

        # Default: proceed to writing
        logger.info("Research appears sufficient, proceeding to writing")
        state.ready_to_write = True
        return "ready_to_write"

    # ========================================================================
    # HELPER METHODS - LLM-based, no external tools
    # ========================================================================

    def _organize_search_results_with_llm(
        self, topic: str, search_results: List[Dict]
    ) -> Dict[str, Any]:
        """Organize search results using LLM reasoning instead of external tool."""

        if not search_results:
            return {
                "categories": {},
                "summary": "No search results to organize",
                "coverage_score": 0.0,
            }

        # Prepare search results for LLM
        results_text = ""
        for i, result in enumerate(search_results[:10]):  # Limit for prompt length
            content = result.get("content", "")[:200]  # Truncate content
            results_text += f"Result {i+1}: {content}...\n"

        # Use template prompt
        prompt = knowledge_organization_prompt(topic=topic, search_results=results_text)

        try:
            organization_response = self.api_client.call_api(prompt)

            # Simple parsing of categories (basic implementation)
            categories = self._parse_organization_response(organization_response)

            return {
                "categories": categories,
                "summary": f"Organized {len(search_results)} search results into {len(categories)} categories",
                "coverage_score": min(
                    1.0, len(search_results) / 10
                ),  # Simple heuristic
            }

        except Exception as e:
            logger.warning(f"LLM knowledge organization failed: {e}")
            return {
                "categories": {"general": search_results},
                "summary": "Organization failed, using general category",
                "coverage_score": 0.5,
            }

    def _parse_organization_response(self, response: str) -> Dict[str, List]:
        """Parse LLM organization response into categories."""
        categories = {}
        current_category = None

        for line in response.split("\n"):
            line = line.strip()
            if line and ":" in line and not line.startswith("-"):
                # This looks like a category header
                category_name = line.split(":")[0].strip()
                current_category = category_name
                categories[current_category] = []
            elif line.startswith("-") and current_category:
                # This is a point under the current category
                point = line[1:].strip()
                if point:
                    categories[current_category].append(
                        {"content": point, "source": "organized"}
                    )

        return categories

    def _parse_outline_response(self, response: str, topic: str) -> Outline:
        """Parse LLM outline response into structured Outline object."""
        lines = response.strip().split("\n")

        title = topic  # Default fallback
        headings = []
        subheadings = {}

        for line in lines:
            line = line.strip()
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line and line[0].isdigit() and "." in line:
                heading = line.split(".", 1)[1].strip()
                headings.append(heading)
                subheadings[heading] = []

        # Fallback outline if parsing fails
        if not headings:
            headings = [
                "Introduction",
                "Background and Context",
                "Key Concepts",
                "Applications and Examples",
                "Current Developments",
                "Future Implications",
            ]
            subheadings = {heading: [] for heading in headings}

        return Outline(title=title, headings=headings, subheadings=subheadings)

    def _generate_research_queries_from_outline(
        self, outline: Outline, topic: str
    ) -> List[str]:
        """Generate targeted research queries based on outline sections."""
        queries = []

        for heading in outline.headings:
            # Main topic query for each section
            queries.append(f"{heading} {topic}")

            # Specific information queries based on heading content
            heading_lower = heading.lower()
            if "introduction" in heading_lower or "background" in heading_lower:
                queries.append(f"overview of {topic}")
            elif "concept" in heading_lower or "definition" in heading_lower:
                queries.append(f"definition {topic}")
            elif "application" in heading_lower or "example" in heading_lower:
                queries.append(f"applications of {topic}")
            elif "current" in heading_lower or "development" in heading_lower:
                queries.append(f"recent developments {topic}")
            elif "future" in heading_lower or "implication" in heading_lower:
                queries.append(f"future of {topic}")

        return queries

    def _format_outline_for_prompt(self, outline: Outline) -> str:
        """Format outline for inclusion in prompts."""
        formatted = f"Title: {outline.title}\n\n"
        for i, heading in enumerate(outline.headings, 1):
            formatted += f"{i}. {heading}\n"
        return formatted

    def _analyze_knowledge_coverage(
        self, outline: Outline, organized_knowledge: Dict
    ) -> Dict[str, float]:
        """Analyze how well each outline section is covered by available knowledge."""
        coverage = {}

        if not organized_knowledge or not organized_knowledge.get("categories"):
            return {heading: 0.0 for heading in outline.headings}

        for heading in outline.headings:
            # Simple heuristic: count relevant results for each section
            relevant_count = 0
            total_count = sum(
                len(results) for results in organized_knowledge["categories"].values()
            )

            if total_count > 0:
                # Look for heading keywords in organized content
                heading_words = heading.lower().split()
                for category, results in organized_knowledge["categories"].items():
                    category_words = category.lower().split("_")

                    # Check for keyword overlap
                    overlap = set(heading_words) & set(category_words)
                    if overlap:
                        relevant_count += len(results)

                coverage[heading] = min(
                    1.0, relevant_count / max(1, total_count / len(outline.headings))
                )
            else:
                coverage[heading] = 0.0

        return coverage

    def _identify_knowledge_gaps(
        self, outline: Outline, organized_knowledge: Dict
    ) -> List[str]:
        """Identify sections with insufficient knowledge coverage."""
        coverage = self._analyze_knowledge_coverage(outline, organized_knowledge)

        gaps = []
        for heading, score in coverage.items():
            if score < self.knowledge_coverage_threshold:
                gaps.append(heading)

        return gaps

    def _generate_section_content(
        self, section_heading: str, topic: str, organized_knowledge: Optional[Dict]
    ) -> str:
        """Generate content for a specific section using organized knowledge."""

        # Find relevant information for this section
        relevant_info = self._find_relevant_info_for_section(
            section_heading, organized_knowledge
        )

        # Choose appropriate template based on available information
        if relevant_info:
            prompt = section_content_prompt_with_research(
                section_heading=section_heading,
                topic=topic,
                relevant_info=relevant_info,
            )
        else:
            prompt = section_content_prompt_without_research(
                section_heading=section_heading, topic=topic
            )

        return self.api_client.call_api(prompt)

    def _find_relevant_info_for_section(
        self, section_heading: str, organized_knowledge: Optional[Dict]
    ) -> str:
        """Find relevant organized information for a specific section."""
        if not organized_knowledge or not organized_knowledge.get("categories"):
            return ""

        relevant_parts = []
        section_words = set(section_heading.lower().split())

        for category, results in organized_knowledge["categories"].items():
            category_words = set(category.lower().split("_"))

            # Check for keyword overlap
            if section_words & category_words:
                for result in results[:2]:  # Limit to top 2 results per category
                    content = result.get("content", "")
                    if content:
                        # Truncate long content
                        if len(content) > 300:
                            content = content[:300] + "..."
                        relevant_parts.append(f"- {content}")

        return "\n".join(relevant_parts[:5])  # Limit total relevant information

    def _create_article_from_state(self, state: WriterState) -> Article:
        """Create Article object from final workflow state."""

        # Use refined outline if available, otherwise initial outline
        final_outline = state.refined_outline or state.initial_outline

        # Extract sections from content
        sections = {}
        if final_outline and state.article_content:
            section_pattern = r"## (.+?)\n\n(.*?)(?=\n## |\Z)"
            matches = re.findall(section_pattern, state.article_content, re.DOTALL)
            for section_title, section_content in matches:
                sections[section_title.strip()] = section_content.strip()

        # Build comprehensive metadata
        metadata = state.metadata.copy()
        metadata.update(
            {
                "research_iterations": state.research_iterations,
                "total_search_results": len(state.search_results),
                "knowledge_coverage": (
                    state.organized_knowledge.get("coverage_score", 0)
                    if state.organized_knowledge
                    else 0
                ),
            }
        )
