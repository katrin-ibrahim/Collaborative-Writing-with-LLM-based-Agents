# src/collaborative/agents/writer_agent.py
"""
Refactored WriterAgent using clean architecture with real tools only.
"""

import logging
import re
from langgraph.graph import END, StateGraph
from typing import Any, Dict

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    chunk_selection_detailed_prompt,
    context_decision_prompt,
    planning_prompt,
    search_query_generation_prompt,
    section_content_prompt_with_research,
    section_content_prompt_without_research,
)
from src.collaborative.memory.memory import SharedMemory
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
        """Build memory-enabled workflow: search → outline → decide_context → tools ↔ write."""
        workflow = StateGraph(SharedMemory)

        # Add workflow nodes
        workflow.add_node("search", self._search_node)
        workflow.add_node("outline", self._outline_node)
        workflow.add_node("decide_context", self._decide_context_node)
        workflow.add_node("tools", self._tools_node)
        workflow.add_node("write", self._write_node)

        # Set entry point to search
        workflow.set_entry_point("search")

        # Search → Outline (always)
        workflow.add_edge("search", "outline")

        # Outline → Decide Context (always)
        workflow.add_edge("outline", "decide_context")

        # Decide Context → Tools or Write (conditional)
        workflow.add_conditional_edges(
            "decide_context",
            self._should_use_tools,
            {"tools": "tools", "write": "write"},
        )

        # Tools → Write (always)
        workflow.add_edge("tools", "write")

        # Write → Tools or End (conditional)
        workflow.add_conditional_edges(
            "write", self._should_continue_writing, {"tools": "tools", "end": END}
        )

        return workflow.compile()

    def process(self) -> None:
        """Process topic through memory-enabled workflow using shared memory."""

        # Get shared memory from ConfigContext
        shared_memory = ConfigContext.get_memory_instance()
        if not shared_memory:
            raise RuntimeError(
                "SharedMemory not available in ConfigContext. Make sure it's initialized before calling WriterAgent.process()"
            )

        topic = shared_memory.state.topic
        logger.info(f"Starting writer workflow for: {topic}")

        # Initialize workflow metadata if not present
        if not shared_memory.state.metadata.get("decisions"):
            shared_memory.state.metadata["decisions"] = []
        if not shared_memory.state.metadata.get("method"):
            shared_memory.state.metadata["method"] = "writer_agent"
        if not shared_memory.state.metadata.get("workflow_version"):
            shared_memory.state.metadata["workflow_version"] = "3.0"

        # Execute workflow
        try:
            self.workflow.invoke(shared_memory)
            # Results are automatically stored in shared_memory by the workflow

            logger.info(
                f"Completed writer workflow for '{topic}' "
                f"({len(shared_memory.state.research_chunks)} research chunks used)"
            )

        except Exception as e:
            logger.error(f"Writer workflow failed for '{topic}': {e}")
            raise

    def _search_node(self, state: SharedMemory) -> Dict[str, Any]:
        """
        Search node:
        1. Direct search on topic to get top result as context
        2. Generate targeted queries using this context
        3. Execute all searches and organize results
        """

        # Use the clean dataclass access
        topic = state.state.topic

        logger.info(f"Starting search for: {topic}")

        # Step 1: Direct search on topic using the tool (it handles storage automatically)
        context = ""
        try:
            direct_search = self.search_tool.invoke(topic)
            if direct_search.get("success") and direct_search.get("chunk_summaries"):
                # Get summaries for context generation
                summaries = direct_search["chunk_summaries"]
                if summaries:
                    context = summaries[0].get("description", "")
                logger.info(
                    f"Direct search stored {direct_search.get('total_chunks', 0)} chunks"
                )
        except Exception as e:
            logger.warning(f"Direct search failed: {e}")

        # Step 2: Generate targeted queries using context
        prompt = search_query_generation_prompt(topic, context, self.num_queries)
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

        # Step 3: Execute targeted queries (tool handles storage automatically)
        total_chunks_stored = direct_search.get("total_chunks", 0)

        for query in queries:
            try:
                search_result = self.search_tool.invoke(query)
                if search_result.get("success"):
                    total_chunks_stored += search_result.get("total_chunks", 0)
                    logger.info(
                        f"Query '{query}' stored {search_result.get('total_chunks', 0)} chunks"
                    )
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        state.state.metadata["decisions"].append(
            f"Completed search: direct + {len(queries)} targeted queries, {total_chunks_stored} total chunks stored"
        )

        logger.info(
            f"Search completed: {len(queries)} targeted queries + direct search, "
            f"{total_chunks_stored} total chunks stored"
        )

        return {}

    def _outline_node(self, state: SharedMemory) -> Dict[str, Any]:
        """Create outline using LLM reasoning and available research."""

        topic = state.state.topic
        logger.info(f"Creating outline for: {topic}")

        # Use template prompt to create outline
        prompt = planning_prompt(topic)
        outline_response = self.api_client.call_api(prompt)
        initial_outline = self.parse_outline(outline_response, topic)

        state.state.initial_outline = initial_outline
        state.state.metadata["decisions"].append(
            f"Created outline with {len(initial_outline.headings)} sections"
        )

        logger.info(f"Created outline with {len(initial_outline.headings)} sections")

        return {}

    def _decide_context_node(self, state: SharedMemory) -> Dict[str, Any]:
        """LLM decides what additional context it needs before writing."""

        logger.info("Deciding what context is needed for writing")

        # Get current context summary
        chunk_summaries = state.get_chunk_summaries()
        context_summary = (
            f"Available chunks: {len(chunk_summaries)}"
            if chunk_summaries
            else "No chunks available"
        )

        # Prepare outline info
        outline_info = ""
        if state.state.initial_outline:
            outline_info = (
                f"Sections to write: {', '.join(state.state.initial_outline.headings)}"
            )

        # Simple decision - ask if tools are needed
        prompt = context_decision_prompt(
            state.state.topic,
            context_summary,
            outline_info,
            len(state.state.research_chunks),
        )

        decision_response = self.api_client.call_api(prompt)

        # Store the decision for the conditional edge
        state.state.metadata["context_decision"] = decision_response.strip().lower()
        state.state.metadata["decisions"].append(
            f"Context decision: {decision_response.strip()}"
        )

        logger.info(f"Context decision: {decision_response.strip()}")

        return {}

    def _tools_node(self, state: SharedMemory) -> Dict[str, Any]:
        """Use tools to gather additional context based on LLM decision."""

        logger.info("Using tools to gather additional context")

        # Select relevant research chunks based on outline
        chunk_summaries = state.get_chunk_summaries()
        if chunk_summaries:
            outline_info = ""
            if state.state.initial_outline:
                outline_info = f"Sections to write: {', '.join(state.state.initial_outline.headings)}"

            # Create chunk selection prompt
            chunks_info = []
            for chunk_id, summary in list(chunk_summaries.items())[
                :10
            ]:  # Limit to 10 for LLM
                chunks_info.append(f"{chunk_id}: {summary}")

            prompt = chunk_selection_detailed_prompt(
                state.state.topic, outline_info, chr(10).join(chunks_info)
            )

            selection_response = self.api_client.call_api(prompt)

            # Parse the selection
            try:
                selection_response = selection_response.strip()
                if selection_response.upper() != "NONE":
                    # Parse comma-separated chunk IDs
                    chunk_ids = [
                        x.strip() for x in selection_response.split(",") if x.strip()
                    ]

                    # Mark selected chunks (for now just log them)
                    selected_chunks = [
                        cid for cid in chunk_ids if cid in chunk_summaries
                    ]

                    state.state.metadata["decisions"].append(
                        f"Selected {len(selected_chunks)} chunks for context: {selected_chunks[:3]}"
                    )

                    logger.info(f"Selected {len(selected_chunks)} chunks for context")
                else:
                    logger.info("No relevant chunks selected")

            except Exception as e:
                logger.warning(f"Failed to parse chunk selection: {e}")

        return {}

    def _write_node(self, state: SharedMemory) -> Dict[str, Any]:
        """Generate article content using LLM with organized knowledge."""

        logger.info("Generating article content")

        working_outline = state.state.initial_outline

        if not working_outline:
            state.state.article_content = (
                f"# {state.state.topic}\n\nUnable to create outline for article."
            )
            return {}

        # Initialize article with title
        article_parts = [f"# {working_outline.title}"]

        # Generate each section using available research chunks
        for section_heading in working_outline.headings:
            try:
                section_content = self._generate_section_content(
                    section_heading, state.state.topic, state
                )

                article_parts.append(f"## {section_heading}")
                article_parts.append(section_content)

            except Exception as e:
                logger.warning(f"Failed to generate section '{section_heading}': {e}")
                article_parts.append(f"## {section_heading}")
                article_parts.append("Content generation failed for this section.")

        # Combine all parts
        full_content = "\n\n".join(article_parts)
        state.state.article_content = full_content

        state.state.metadata["decisions"].append(
            f"Generated article with {len(working_outline.headings)} sections"
        )

        logger.info(
            f"Article generation completed: {len(full_content)} characters, "
            f"{len(working_outline.headings)} sections"
        )

        return {}

    # ========================================================================
    # CONDITIONAL EDGE FUNCTIONS
    # ========================================================================

    def _should_use_tools(self, state: SharedMemory) -> str:
        """Determine if agent should use tools or proceed to writing."""
        decision = state.state.metadata.get("context_decision", "").lower()

        # Simple decision logic based on LLM response
        if any(
            word in decision
            for word in ["yes", "need", "require", "fetch", "get", "retrieve"]
        ):
            return "tools"
        else:
            return "write"

    def _should_continue_writing(self, state: SharedMemory) -> str:
        """Determine if agent should get more context or finish."""
        # For now, simple logic - finish after first write
        # Later we can add more sophisticated logic based on:
        # - Content quality assessment
        # - Missing information detection
        # - Feedback from previous iterations

        # Check if we have substantial content
        if len(state.state.article_content) > 500:  # Basic length check
            return "end"
        else:
            return "tools"  # Get more context if content seems insufficient

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
        self, section_heading: str, topic: str, memory_state: SharedMemory
    ) -> str:
        """Generate content for a specific section using research chunks."""

        # Find relevant information for this section from research chunks
        relevant_info = self._find_relevant_info_for_section(
            section_heading, memory_state
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
        self, section_heading: str, memory_state: SharedMemory
    ) -> str:
        """Find relevant information for section from research chunks."""

        chunks = memory_state.state.research_chunks
        if not chunks:
            return ""

        relevant_parts = []

        # Simple relevance check based on keyword overlap
        section_keywords = set(section_heading.lower().split())

        scored_chunks = []
        for chunk_id, chunk_data in chunks.items():
            content = chunk_data.get("content", "")
            description = chunk_data.get("description", "")

            if content:
                content_words = set(content.lower().split())
                desc_words = set(description.lower().split())
                all_words = content_words | desc_words

                overlap = len(section_keywords & all_words)
                scored_chunks.append((overlap, chunk_data))

        # Sort by relevance and take top results
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        for score, chunk_data in scored_chunks[:4]:
            content = chunk_data.get("content", "")
            if content and score > 0:  # Only include chunks with some relevance
                if len(content) > 300:
                    content = content[:300] + "..."
                relevant_parts.append(f"- {content}")

        return "\n".join(relevant_parts[:5])

    def _create_article_from_state(self, state: SharedMemory) -> Article:
        """Create Article object from final workflow state."""

        # Use initial outline since we no longer refine
        final_outline = state.state.initial_outline

        # Extract sections from content
        sections = {}
        if final_outline and state.state.article_content:
            section_pattern = r"## (.+?)\n\n(.*?)(?=\n## |\Z)"
            matches = re.findall(
                section_pattern, state.state.article_content, re.DOTALL
            )
            for section_title, section_content in matches:
                sections[section_title.strip()] = section_content.strip()

        # Build comprehensive metadata
        metadata = state.state.metadata.copy()
        metadata.update(
            {
                "total_research_chunks": len(state.state.research_chunks),
            }
        )

        # CREATE AND RETURN THE ARTICLE
        article = Article(
            title=final_outline.title if final_outline else state.state.topic,
            content=state.state.article_content,
            outline=final_outline,
            sections=sections,
            metadata=metadata,
        )

        return article
