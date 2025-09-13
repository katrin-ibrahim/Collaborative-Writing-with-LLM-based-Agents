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
from src.collaborative.data_models import ReviewFeedback, WriterState
from src.collaborative.tools.writer_toolkit import WriterToolkit
from src.config.config_context import ConfigContext
from src.utils.data import Article, Outline

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
                "workflow_version": "1.0",
                "tool_usage": [],
                "decisions": [],
            },
        )

        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            final_state = WriterState(**final_state)
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

        # Limit queries per iteration for efficiency
        current_queries = state.research_queries[
            : self.max_research_queries_per_iteration
        ]

        new_results = []
        query_result_mapping = []  # Track which results came from which queries

        for query in current_queries:
            try:
                search_result = self.search_tool.invoke(query)
                if search_result.get("success") and search_result.get("results"):
                    for result in search_result["results"]:
                        query_result_mapping.append((query, result))
                    new_results.extend(search_result["results"])
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # Add new results to existing ones
        state.search_results.extend(new_results)

        # Organize search results (do not pass query_result_mapping if not needed)
        state.organized_knowledge = self._organize_search_results(
            state.initial_outline, query_result_mapping
        )

        return state

    def _organize_search_results(
        self, outline: Outline, query_result_mapping: List[tuple] = None
    ) -> Dict:
        if not query_result_mapping or not outline:
            return {"by_section": {}, "cross_cutting": []}

        organized = {
            "by_section": {
                section: {"primary_results": [], "supporting_results": []}
                for section in outline.headings
            },
            "cross_cutting": [],
            "metadata": {},
        }

        # Step 1: Group results by query (section)
        query_groups = {}
        for query, result in query_result_mapping:
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append(result)

        # Step 2: Assign primary results - each section gets its own query results
        section_to_results = {}
        for section_index, section in enumerate(outline.headings):
            if section_index < len(query_groups):
                query = list(query_groups.keys())[section_index]
                section_results = query_groups[query]
                section_to_results[section] = section_results

                # All results from this section's query are primary for this section
                for result in section_results:
                    relevance = result.get("relevance_score", 0.5)
                    if relevance >= 0.7:
                        organized["by_section"][section]["primary_results"].append(
                            result
                        )
                    else:
                        organized["by_section"][section]["supporting_results"].append(
                            result
                        )

        # Step 3: Cross-pollinate - check if results from other sections are relevant
        for target_section in outline.headings:
            for other_section in outline.headings:
                if target_section == other_section:
                    continue  # Skip same section

                other_results = section_to_results.get(other_section, [])
                for result in other_results:
                    # Check if this result is relevant to the target section
                    if self._is_result_relevant_to_section(result, target_section):
                        # Add as supporting result (avoid duplicates)
                        if (
                            result
                            not in organized["by_section"][target_section][
                                "supporting_results"
                            ]
                        ):
                            organized["by_section"][target_section][
                                "supporting_results"
                            ].append(result)

        organized["metadata"] = {
            "total_results": len(query_result_mapping),
            "unique_sources": len(
                set(r.get("source", "") for _, r in query_result_mapping)
            ),
            "cross_cutting_count": len(organized["cross_cutting"]),
        }

        return organized

    def _is_result_relevant_to_section(self, result: Dict, section_name: str) -> bool:
        """Check if a result is relevant to a specific section."""
        content = result.get("content", "")
        if not content:
            return False

        content_words = set(content.lower().split())
        section_words = set(section_name.lower().split())

        # Simple word overlap - could be enhanced with embeddings
        overlap = len(content_words & section_words)
        len(content_words)

        # Relevance based on overlap ratio and minimum relevance score
        overlap_ratio = overlap / max(len(section_words), 1)
        min_relevance = result.get("relevance_score", 0) >= 0.5

        return overlap_ratio > 0.3 and min_relevance  # Adjust thresholds as needed

    def _generate_content_summary_for_refinement(
        self, organized_knowledge: Dict
    ) -> str:
        """Generate a content summary from top search results for outline refinement."""
        if not organized_knowledge:
            return "No research content available"

        summary_parts = []

        # Get top results from each section
        for section, data in organized_knowledge.get("by_section", {}).items():
            primary_results = data.get("primary_results", [])
            supporting_results = data.get("supporting_results", [])

            # Combine and sort by relevance
            all_section_results = primary_results + supporting_results
            all_section_results.sort(
                key=lambda x: x.get("relevance_score", 0), reverse=True
            )

            top_results = all_section_results[:3]  # Top 3 for this section

            if top_results:
                summary_parts.append(f"\n{section}:")
                for i, result in enumerate(top_results, 1):
                    content = result.get("content", "")
                    # Truncate long content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    summary_parts.append(f"  {i}. {content}")

        # Add cross-cutting content if any
        cross_cutting = organized_knowledge.get("cross_cutting", [])
        if cross_cutting:
            summary_parts.append(f"\nOther relevant findings:")
            for i, result in enumerate(cross_cutting[:2], 1):  # Top 2 cross-cutting
                content = result.get("content", "")
                if len(content) > 500:
                    content = content[:500] + "..."
                summary_parts.append(f"  {i}. {content}")

        return "\n".join(summary_parts)

    def _refine_outline_node(self, state: WriterState) -> WriterState:
        """Refine outline based on research findings using LLM reasoning."""

        logger.info("Refining outline based on research findings")

        if not state.organized_knowledge:
            # No knowledge to work with, keep original outline
            state.refined_outline = state.initial_outline
            state.ready_to_write = True
            return state

        # Use template prompt for refinement
        content_summary_for_refinement = self._generate_content_summary_for_refinement(
            state.organized_knowledge
        )

        prompt = refinement_prompt(
            topic=state.topic,
            current_outline=self._format_outline_for_prompt(state.initial_outline),
            content_summary=content_summary_for_refinement,
        )

        refined_response = self.api_client.call_api(prompt)
        refined_outline = self.parse_outline(refined_response, state.topic)

        state.refined_outline = refined_outline

        return state

    def _generate_knowledge_summary(self, organized_knowledge: Dict) -> str:
        """Generate a summary of organized knowledge for prompting."""
        if not organized_knowledge:
            return "No research data available"

        metadata = organized_knowledge.get("metadata", {})
        cross_cutting = organized_knowledge.get("cross_cutting", [])

        summary_parts = []
        summary_parts.append(
            f"Total research results: {metadata.get('total_results', 0)}"
        )
        summary_parts.append(f"Unique sources: {metadata.get('unique_sources', 0)}")

        if cross_cutting:
            summary_parts.append(f"Cross-cutting results: {len(cross_cutting)}")

        return "\n".join(summary_parts)

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

        new_metadata = state.metadata.copy()
        # Combine all parts
        full_content = "\n\n".join(article_parts)
        state.article_content = full_content

        new_metadata["decisions"].append(
            f"Generated article with {len(working_outline.headings)} sections"
        )
        state.metadata = new_metadata

        logger.info(
            f"Article generation completed: {len(full_content)} characters, "
            f"{len(working_outline.headings)} sections"
        )

        return state

    def _decide_after_outline_refinement(self, state: WriterState) -> str:
        """Decide whether to research more or proceed to writing."""

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
            }
        )

        # CREATE AND RETURN THE ARTICLE - this was missing!
        article = Article(
            title=final_outline.title if final_outline else state.topic,
            content=state.article_content,
            outline=final_outline,
            sections=sections,
            metadata=metadata,
        )

        return article
