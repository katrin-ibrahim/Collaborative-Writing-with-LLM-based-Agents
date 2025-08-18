# src/agents/writer/writer_agent.py
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from src.agents.base_agent import BaseAgent
from src.agents.tools.content_toolkit import generate_outline, generate_section_content
from src.agents.tools.search_toolkit import create_search_tool
from src.utils.data import Article, Outline, SearchResult


class WriterState(TypedDict):
    """Simplified state for 3-node writer workflow."""

    messages: Annotated[List[BaseMessage], operator.add]
    topic: str
    search_results: List[SearchResult]
    outline: Optional[Outline]
    article_content: str
    research_iterations: int
    needs_more_research: bool
    metadata: Dict[str, Any]


class WriterAgent(BaseAgent):
    """
    Simplified 3-node Writer agent using LangGraph with autonomous tool selection.

    Workflow: research -> plan_outline -> write_content
    Features conditional research-more loop and baseline format compatibility.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Simplified configuration
        self.max_research_iterations = config.get("writer.max_research_iterations", 2)
        self.use_external_knowledge = config.get("writer.use_external_knowledge", True)

        # Create search tool with retrieval config from CLI args
        retrieval_config = config.get("retrieval_config")
        self.search_tool = create_search_tool(retrieval_config)

        self.workflow = self._build_workflow()

        self.logger.info(
            f"WriterAgent initialized with max_research_iterations={self.max_research_iterations}"
        )

    def _build_workflow(self) -> StateGraph:
        """Build simplified 3-node LangGraph workflow."""
        workflow = StateGraph(WriterState)

        # Add 3 core nodes
        workflow.add_node("research", self._research_node)
        workflow.add_node("plan_outline", self._plan_outline_node)
        workflow.add_node("write_content", self._write_content_node)

        # Set entry point
        workflow.set_entry_point("research")

        # Add conditional edges
        workflow.add_conditional_edges(
            "research",
            self._decide_after_research,
            {"research_more": "research", "plan_outline": "plan_outline"},
        )

        workflow.add_edge("plan_outline", "write_content")
        workflow.add_edge("write_content", END)

        return workflow.compile()

    def process(self, topic: str) -> Article:
        """Process topic through simplified 3-node workflow."""
        initial_state = WriterState(
            messages=[
                HumanMessage(content=f"Write a comprehensive article about: {topic}")
            ],
            topic=topic,
            search_results=[],
            outline=None,
            article_content="",
            research_iterations=0,
            needs_more_research=True,
            metadata={"method": "simplified_writer", "decisions": []},
        )

        final_state = self.workflow.invoke(initial_state)
        return self._state_to_article(final_state)

    def _research_node(self, state: WriterState) -> WriterState:
        """Research node with autonomous tool selection."""
        if not self.use_external_knowledge:
            state["needs_more_research"] = False
            state["metadata"]["decisions"].append(
                "Skipped research (external knowledge disabled)"
            )
            return state

        if state["research_iterations"] >= self.max_research_iterations:
            state["needs_more_research"] = False
            state["metadata"]["decisions"].append(
                f"Max research iterations reached ({self.max_research_iterations})"
            )
            return state

        # LLM generates focused research queries for the specific topic
        research_prompt = f"""
        I need to research "{state['topic']}" for writing a comprehensive Wikipedia-style article.

        Current status:
        - Research iteration: {state['research_iterations'] + 1}/{self.max_research_iterations}
        - Existing results: {len(state['search_results'])}

        Analyze the topic "{state['topic']}" and generate 2-3 specific search queries that will help me find information about this topic.

        Break down the topic into its key components and generate targeted search queries.


        Generate focused search queries that will help me understand this specific topic:

        Format: one query per line, no numbers or bullets.
        """

        response = self.api_client.call_api(research_prompt)
        search_queries = [
            q.strip()
            for q in response.split("\n")
            if q.strip() and not q.strip().startswith(("1.", "2.", "3.", "-", "*"))
        ][  # todo: remove hardcoded limit
            :3
        ]  # Limit to 3 queries

        # Execute searches with focused approach
        new_results = []

        # Execute each LLM-generated query
        for query in search_queries:
            try:
                result = self.search_tool(
                    {"query": query, "wiki_results": 3, "web_results": 2}
                )
                search_results = [SearchResult(**r) for r in result["results"]]
                new_results.extend(search_results)
                self.logger.info(
                    f"Search successful for query: '{query}' - found {len(search_results)} results"
                )
            except Exception as e:
                self.logger.warning(f"Search failed for query '{query}': {e}")
                # Continue with other queries

        # Only add results if we found any
        if new_results:
            state["search_results"].extend(new_results)
            state["metadata"]["decisions"].append(
                f"Research iteration {state['research_iterations'] + 1}: Found {len(new_results)} results"
            )
        else:
            state["metadata"]["decisions"].append(
                f"Research iteration {state['research_iterations'] + 1}: No results found, will use internal knowledge"
            )

        state["research_iterations"] += 1

        return state

    def _plan_outline_node(self, state: WriterState) -> WriterState:
        """Plan outline node using research results."""
        # Create context from search results
        context = ""
        if state["search_results"]:
            context_parts = []
            for i, result in enumerate(state["search_results"][:8]):
                context_parts.append(f"[Source {i+1}]: {result.content}")
            context = "\n\n".join(context_parts)[:1500]  # Limit context length

        # Generate outline using tool
        outline_result = generate_outline.invoke(
            {"topic": state["topic"], "context": context}
        )

        # Convert to Outline object
        outline = Outline(
            title=outline_result["title"],
            headings=outline_result["headings"],
            subheadings=outline_result["subheadings"],
        )
        state["outline"] = outline

        state["metadata"]["decisions"].append(
            f"Created outline with {len(outline.headings)} sections"
        )
        return state

    def _write_content_node(self, state: WriterState) -> WriterState:
        """Write content node with autonomous section generation."""
        if not state["outline"]:
            state["article_content"] = (
                f"# {state['topic']}\n\nUnable to generate outline for comprehensive article."
            )
            return state

        # Initialize article
        state["article_content"] = f"# {state['outline'].title}\n\n"

        # Create context from all search results
        context = ""
        if state["search_results"]:
            context_parts = []
            for i, result in enumerate(state["search_results"]):
                context_parts.append(f"[Source {i+1}]: {result.content}")
            context = "\n\n".join(context_parts)[:2000]  # Limit context length

        # Write each section using tool
        for section_heading in state["outline"].headings:
            # Generate section content using tool
            content_result = generate_section_content.invoke(
                {
                    "section_title": section_heading,
                    "context": context,
                    "article_topic": state["topic"],
                    "target_length": "300-400",
                }
            )

            # Call LLM to generate actual content
            section_content = self.api_client.call_api(
                content_result["generation_prompt"]
            )

            state[
                "article_content"
            ] += f"## {section_heading}\n\n{section_content.strip()}\n\n"

        state["metadata"]["decisions"].append(
            f"Generated {len(state['outline'].headings)} sections"
        )
        return state

    def _decide_after_research(self, state: WriterState) -> str:
        """Conditional logic for research-more loop with topic validation."""
        # Check iteration limit
        if state["research_iterations"] >= self.max_research_iterations:
            return "plan_outline"

        # Evaluate if we have enough information about the specific topic
        content_sample = ""
        if state["search_results"]:
            content_sample = " ".join(
                [r.content[:200] for r in state["search_results"][:3]]
            )

        # LLM decides if more research is needed
        decision_prompt = f"""
        I've researched "{state['topic']}" and found {len(state['search_results'])} results.
        Research iteration: {state['research_iterations']}/{self.max_research_iterations}

        Sample of found content:
        {content_sample[:500]}...

        Evaluate if I have sufficient information to write a comprehensive article about this specific topic:

        Questions to consider:
        1. Do the search results contain information specifically about "{state['topic']}"?
        2. Is the information detailed enough to write a comprehensive article?
        3. Are there key aspects of this topic that seem to be missing?
        4. Would I benefit from additional targeted searches?

        Respond with just: RESEARCH_MORE or PLAN_OUTLINE
        """

        response = self.api_client.call_api(decision_prompt).strip().upper()

        if (
            "RESEARCH_MORE" in response
            and state["research_iterations"] < self.max_research_iterations
        ):
            state["metadata"]["decisions"].append(
                "LLM decided: need more specific research"
            )
            return "research_more"

        return "plan_outline"

    def _state_to_article(self, state: WriterState) -> Article:
        """Convert final state to baseline-compatible Article format."""
        # Extract sections from article content
        sections = {}
        if state["outline"]:
            import re

            section_pattern = r"## (.+?)\n\n(.*?)(?=\n## |\Z)"
            matches = re.findall(section_pattern, state["article_content"], re.DOTALL)
            for section_title, section_content in matches:
                sections[section_title.strip()] = section_content.strip()

        # Build metadata compatible with baseline format
        metadata = state["metadata"].copy()
        metadata.update(
            {
                "method": "simplified_writer",
                "word_count": len(state["article_content"].split()),
                "total_sections": len(sections),
                "research_iterations": state["research_iterations"],
                "total_search_results": len(state["search_results"]),
                "sources": [result.source for result in state["search_results"]],
            }
        )

        return Article(
            title=state["outline"].title if state["outline"] else state["topic"],
            content=state["article_content"],
            outline=state["outline"],
            sections=sections,
            metadata=metadata,
        )
