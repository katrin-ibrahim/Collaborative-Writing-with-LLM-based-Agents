# src/collaborative/agents/writer_agent.py
"""
Refactored WriterAgent using clean architecture with real tools only.
"""

import logging
import re
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from typing import Dict, List, Optional

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    improvement_prompt,
    planning_prompt,
    refinement_prompt,
    section_content_prompt_with_research,
    section_content_prompt_without_research,
)
from src.collaborative.data_models import Outline, ReviewFeedback, WriterState
from src.collaborative.tools.writer_toolkit import WriterToolkit
from src.config.config_context import ConfigContext
from src.utils.data import Article

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """
    Sophisticated Writer agent with planning-first workflow.

    Workflow: plan_outline → targeted_research → refine_outline → write_content
    Uses only real tools (search_and_retrieve) and LLM reasoning.
    """

    def __init__(self):
        super().__init__()
        self.collaboration_config = ConfigContext.get_collaboration_config()
        self.retrieval_config = ConfigContext.get_retrieval_config()

        # Get configuration values with proper defaults
        self.max_research_iterations = getattr(
            self.collaboration_config, "writer.max_research_iterations", 3
        )
        self.knowledge_coverage_threshold = getattr(
            self.collaboration_config, "writer.knowledge_coverage_threshold", 0.7
        )
        self.max_research_queries_per_iteration = getattr(
            self.collaboration_config, "writer.max_queries_per_iteration", 6
        )
        self.max_search_results = getattr(
            self.collaboration_config, "writer.max_search_results", 5
        )
        self.rm_type = getattr(self.retrieval_config, "retrieval_manager", "wiki")

        # Initialize writer toolkit (only search_and_retrieve tool)
        self.toolkit = WriterToolkit(self.retrieval_config)

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
        initial_outline = self.parse_outline(outline_response, state.topic)

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

        # # Limit queries per iteration for efficiency
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

        state.organized_knowledge = self._organize_search_results(
            state.search_results, state.initial_outline, state.research_queries
        )

        return state

    def _organize_search_results(
        self, search_results: List[Dict], outline: Outline, queries: List[str]
    ) -> Dict:
        """Organize search results using hybrid section + relevance approach."""
        if not search_results or not outline:
            return {
                "by_section": {},
                "cross_cutting": [],
                "metadata": {"coverage_score": 0.0},
            }

        # Map queries to sections
        query_to_section = {}
        for i, query in enumerate(queries):
            if i < len(outline.headings):
                query_to_section[query] = outline.headings[i]

        # Initialize organized structure
        organized = {
            "by_section": {
                section: {"primary_results": [], "supporting_results": []}
                for section in outline.headings
            },
            "cross_cutting": [],
            "metadata": {},
        }

        # Organize results by relevance tiers
        for result in search_results:
            relevance = result.get("relevance_score", 0.5)
            result_query = result.get("query", "")
            target_section = query_to_section.get(result_query)

            if target_section and target_section in organized["by_section"]:
                if relevance >= 0.7:
                    organized["by_section"][target_section]["primary_results"].append(
                        result
                    )
                elif relevance >= 0.4:
                    organized["by_section"][target_section][
                        "supporting_results"
                    ].append(result)
            else:
                if relevance >= 0.6:
                    organized["cross_cutting"].append(result)

        # Calculate coverage scores
        section_coverage = {}
        total_coverage = 0
        for section in outline.headings:
            primary_count = len(organized["by_section"][section]["primary_results"])
            supporting_count = len(
                organized["by_section"][section]["supporting_results"]
            )
            section_score = min(
                1.0, (primary_count * 0.7 + supporting_count * 0.3) / 2.0
            )
            section_coverage[section] = section_score
            total_coverage += section_score

        organized["metadata"] = {
            "coverage_score": (
                total_coverage / len(outline.headings) if outline.headings else 0.0
            ),
            "section_coverage": section_coverage,
        }

        return organized

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
        knowledge_summary = self._generate_knowledge_summary(state.organized_knowledge)
        coverage_analysis = self._format_coverage_analysis(state.organized_knowledge)

        prompt = refinement_prompt(
            topic=state.topic,
            current_outline=self._format_outline_for_prompt(state.initial_outline),
            knowledge_summary=knowledge_summary,
            coverage_analysis=coverage_analysis,
        )

        refined_response = self.api_client.call_api(prompt)
        refined_outline = self.parse_outline(refined_response, state.topic)

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

    def _generate_knowledge_summary(self, organized_knowledge: Dict) -> str:
        """Generate a summary of organized knowledge for prompting."""
        if not organized_knowledge:
            return "No research data available"

        metadata = organized_knowledge.get("metadata", {})
        by_section = organized_knowledge.get("by_section", {})
        cross_cutting = organized_knowledge.get("cross_cutting", [])

        summary_parts = []
        summary_parts.append(
            f"Total research results: {metadata.get('total_results', 0)}"
        )
        summary_parts.append(f"Unique sources: {metadata.get('unique_sources', 0)}")
        summary_parts.append(
            f"Overall coverage score: {metadata.get('coverage_score', 0):.2f}"
        )

        if cross_cutting:
            summary_parts.append(f"Cross-cutting results: {len(cross_cutting)}")

        summary_parts.append("\nSection-specific coverage:")
        for section, data in by_section.items():
            primary = len(data.get("primary_results", []))
            supporting = len(data.get("supporting_results", []))
            summary_parts.append(
                f"- {section}: {primary} primary, {supporting} supporting results"
            )

        return "\n".join(summary_parts)

    def _format_coverage_analysis(self, organized_knowledge: Dict) -> str:
        """Format coverage analysis for prompting."""
        if not organized_knowledge:
            return "No coverage analysis available"

        section_coverage = organized_knowledge.get("metadata", {}).get(
            "section_coverage", {}
        )

        analysis_parts = []
        for section, score in section_coverage.items():
            status = (
                "Well covered"
                if score >= 0.6
                else "Needs attention" if score >= 0.3 else "Poorly covered"
            )
            analysis_parts.append(f"- {section}: {score:.2f} ({status})")

        return "\n".join(analysis_parts)

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
            coverage_score = state.organized_knowledge.get("metadata", {}).get(
                "coverage_score", 0
            )

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

    def parse_outline(self, response: str, topic: str):
        lines = response.strip().split("\n")

        title = topic  # fallback
        headings = []
        subheadings = {}

        for line in lines:
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):  # H1 title
                title = line.replace("#", "").strip()
            elif line.startswith("## "):  # H2 headings
                heading = line.replace("##", "").strip()
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
            # TODO: LLM could be used here for more sophisticated query generation
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
        if not organized_knowledge or not organized_knowledge.get("metadata"):
            return {heading: 0.0 for heading in outline.headings}

        return organized_knowledge["metadata"].get("section_coverage", {})

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
        """Find relevant organized information using hybrid structure."""
        if not organized_knowledge or not organized_knowledge.get("by_section"):
            return ""

        section_data = organized_knowledge["by_section"].get(section_heading, {})
        relevant_parts = []

        # Start with primary results (high confidence)
        for result in section_data.get("primary_results", [])[:3]:
            content = result.get("content", "")
            if content:
                if len(content) > 300:
                    content = content[:300] + "..."
                relevant_parts.append(f"- {content}")

        # Add supporting results if needed
        if len(relevant_parts) < 2:
            for result in section_data.get("supporting_results", [])[:2]:
                content = result.get("content", "")
                if content:
                    if len(content) > 300:
                        content = content[:300] + "..."
                    relevant_parts.append(f"- {content}")

        return "\n".join(relevant_parts[:5])

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
