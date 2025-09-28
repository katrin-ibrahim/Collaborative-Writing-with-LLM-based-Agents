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
    search_query_generation_prompt,
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
    Sophisticated Writer agent with simplified search-first workflow.

    Workflow: search → outline → write
    Search includes: direct topic search + context-aware targeted queries
    Uses only real tools (search_and_retrieve) and LLM reasoning.
    """

    def __init__(self):
        super().__init__()
        self.collaboration_config = ConfigContext.get_collaboration_config()
        self.retrieval_config = ConfigContext.get_retrieval_config()

        # Get configuration values with proper defaults
        self.num_queries = getattr(self.retrieval_config, "num_queries", 5)
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
            f"WriterAgent initialized with simplified workflow: search → outline → write "
            f"(max_queries={self.num_queries})"
        )

    def _build_workflow(self) -> StateGraph:
        """Build simplified workflow: search → outline → write."""
        workflow = StateGraph(WriterState)

        # Add workflow nodes
        workflow.add_node("search", self._search_node)
        workflow.add_node("outline", self._outline_node)
        workflow.add_node("write", self._write_node)

        # Set entry point to search
        workflow.set_entry_point("search")

        # Search → Outline (always)
        workflow.add_edge("search", "outline")

        # Outline → Write (always)
        workflow.add_edge("outline", "write")

        # Write → End
        workflow.add_edge("write", END)

        return workflow.compile()

    def process(
        self,
        topic: str,
        previous_article: Optional[Article] = None,
        review_feedback: Optional[ReviewFeedback] = None,
    ) -> Article:
        """Process topic through simplified search → outline → write workflow."""

        logger.info(f"Starting writer workflow for: {topic}")

        # Handle improvement case
        if previous_article and review_feedback:
            return self._improve_article(previous_article, review_feedback, topic)

        # Initialize state for new article
        initial_state = WriterState(
            messages=[HumanMessage(content=f"Write an article about: {topic}")],
            topic=topic,
            research_queries=[],
            search_results=[],
            organized_knowledge=None,
            initial_outline=None,
            article_content="",
            metadata={
                "method": "writer_agent",
                "workflow_version": "2.0",
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
                f"({len(final_state.search_results)} sources)"
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

    def _search_node(self, state: WriterState) -> WriterState:
        """
        Search node:
        1. Direct search on topic to get top result as context
        2. Generate targeted queries using this context
        3. Execute all searches and organize results
        """

        logger.info(f"Starting search for: {state.topic}")

        # Step 1: Direct search on topic to get context
        context = ""
        try:
            direct_search = self.search_tool.invoke(state.topic)
            if direct_search.get("success") and direct_search.get("results"):
                # Get the top result as context
                top_result = direct_search["results"][0]
                context = top_result.get("content", "")
                logger.info(f"Got context from direct search: {context}")
        except Exception as e:
            logger.warning(f"Direct search failed: {e}")

        # Step 2: Generate targeted queries using context
        prompt = search_query_generation_prompt(state.topic, context, self.num_queries)
        queries_response = self.api_client.call_api(prompt)

        # Parse queries from response with basic cleanup
        queries = []
        for line in queries_response.strip().split("\n"):
            line = line.strip()
            if line:
                # Basic cleanup: remove common prefixes and formatting
                if line.lower().startswith(("here are", "query:", "search:")):
                    continue
                # Remove numbering and quotes
                import re

                line = re.sub(
                    r"^\d+[\.\)\:]\s*", "", line
                )  # Remove "1. " or "1) " or "1: "
                line = line.strip("\"'")  # Remove quotes
                # Skip if too short after cleanup
                if len(line.strip()) >= 3:
                    queries.append(line.strip())

        # Limit queries to reasonable number
        queries = queries[: self.num_queries]

        # Step 3: Execute all searches
        all_results = []
        query_result_mapping = []

        # Include direct search results if we have them
        if context:
            direct_results = direct_search.get("results", [])
            all_results.extend(direct_results)
            for result in direct_results:
                query_result_mapping.append((state.topic, result))

        # Execute targeted queries
        for query in queries:
            try:
                search_result = self.search_tool.invoke(query)
                if search_result.get("success") and search_result.get("results"):
                    for result in search_result["results"]:
                        query_result_mapping.append((query, result))
                    all_results.extend(search_result["results"])
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # Store results
        state.research_queries = [state.topic] + queries  # Include direct search
        state.search_results = all_results

        # We can't organize by outline sections yet since we don't have an outline
        # So we'll organize by query instead
        state.organized_knowledge = self._organize_search_results_by_query(
            query_result_mapping
        )

        state.metadata["decisions"].append(
            f"Completed search: direct + {len(queries)} targeted queries, {len(all_results)} total results"
        )

        logger.info(
            f"Search completed: {len(queries)} targeted queries + direct search, "
            f"{len(all_results)} total results"
        )

        return state

    def _outline_node(self, state: WriterState) -> WriterState:
        """Create outline using LLM reasoning and available research."""

        logger.info(f"Creating outline for: {state.topic}")

        # Use template prompt to create outline
        prompt = planning_prompt(state.topic)
        outline_response = self.api_client.call_api(prompt)
        initial_outline = self.parse_outline(outline_response, state.topic)

        state.initial_outline = initial_outline
        state.metadata["decisions"].append(
            f"Created outline with {len(initial_outline.headings)} sections"
        )

        logger.info(f"Created outline with {len(initial_outline.headings)} sections")

        return state

    def _organize_search_results_by_query(
        self, query_result_mapping: List[tuple]
    ) -> Dict:
        """Organize search results by query since we don't have outline sections yet."""
        if not query_result_mapping:
            return {"by_query": {}, "all_results": []}

        organized = {"by_query": {}, "all_results": [], "metadata": {}}

        # Group results by query
        for query, result in query_result_mapping:
            if query not in organized["by_query"]:
                organized["by_query"][query] = []
            organized["by_query"][query].append(result)
            organized["all_results"].append(result)

        # Add metadata
        organized["metadata"] = {
            "total_results": len(query_result_mapping),
            "unique_queries": len(organized["by_query"]),
            "unique_sources": len(
                set(r.get("source", "") for _, r in query_result_mapping)
            ),
        }

        return organized

    def _write_node(self, state: WriterState) -> WriterState:
        """Generate article content using LLM with organized knowledge."""

        logger.info("Generating article content")

        working_outline = state.initial_outline

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
                # Avoid duplicates and limit to reasonable number
                if heading not in headings and len(headings) < 8:
                    headings.append(heading)
                    subheadings[heading] = []

        # Fallback outline if parsing fails or too few sections
        if len(headings) < 3:
            headings = [
                "Introduction",
                "Background and Context",
                "Key Concepts",
                "Applications and Examples",
                "Current Developments",
            ]
            subheadings = {heading: [] for heading in headings}

        return Outline(title=title, headings=headings, subheadings=subheadings)

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
        """Find relevant information for section from all search results."""
        if not organized_knowledge:
            return ""

        relevant_parts = []

        # Check if we have by_section organization (old format) or by_query (new format)
        if organized_knowledge.get("by_section"):
            # Old format - use existing logic
            section_data = organized_knowledge["by_section"].get(section_heading, {})
            for result in section_data.get("primary_results", [])[:3]:
                content = result.get("content", "")
                if content:
                    if len(content) > 300:
                        content = content[:300] + "..."
                    relevant_parts.append(f"- {content}")

            if len(relevant_parts) < 2:
                for result in section_data.get("supporting_results", [])[:2]:
                    content = result.get("content", "")
                    if content:
                        if len(content) > 300:
                            content = content[:300] + "..."
                        relevant_parts.append(f"- {content}")

        elif organized_knowledge.get("all_results"):
            # New format - search through all results for relevance
            all_results = organized_knowledge["all_results"]

            # Simple relevance check based on keyword overlap
            section_keywords = set(section_heading.lower().split())

            scored_results = []
            for result in all_results:
                content = result.get("content", "")
                if content:
                    content_words = set(content.lower().split())
                    overlap = len(section_keywords & content_words)
                    relevance_score = result.get("relevance_score", 0.5)
                    combined_score = overlap + relevance_score
                    scored_results.append((combined_score, result))

            # Sort by relevance and take top results
            scored_results.sort(key=lambda x: x[0], reverse=True)

            for score, result in scored_results[:4]:
                content = result.get("content", "")
                if content:
                    if len(content) > 300:
                        content = content[:300] + "..."
                    relevant_parts.append(f"- {content}")

        return "\n".join(relevant_parts[:5])

    def _create_article_from_state(self, state: WriterState) -> Article:
        """Create Article object from final workflow state."""

        # Use initial outline since we no longer refine
        final_outline = state.initial_outline

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
