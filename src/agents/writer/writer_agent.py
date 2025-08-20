# src/agents/writer/writer_agent.py
"""
Refactored WriterAgent using real tools and sophisticated planning-first workflow.
Eliminates fake ContentToolkit and implements proper tool usage.
"""

import operator

import logging
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from src.agents.base_agent import BaseAgent
from src.agents.tools.agent_toolkit import AgentToolkit
from src.utils.data.models import Article, Outline

logger = logging.getLogger(__name__)


class WriterState(TypedDict):
    """State for sophisticated writer workflow with planning-first approach."""

    messages: Annotated[List[BaseMessage], operator.add]
    topic: str

    # Planning phase
    initial_outline: Optional[Outline]
    research_queries: List[str]

    # Research phase
    search_results: List[Dict[str, Any]]
    organized_knowledge: Optional[Dict[str, Any]]

    # Refinement phase
    refined_outline: Optional[Outline]
    knowledge_gaps: List[str]

    # Writing phase
    article_content: str
    self_validation: Optional[Dict[str, Any]]

    # Flow control
    research_iterations: int
    ready_to_write: bool

    # Metadata
    metadata: Dict[str, Any]


class WriterAgent(BaseAgent):
    """
    Sophisticated Writer agent with planning-first workflow and real tool usage.

    Workflow: plan_outline → targeted_research → refine_outline → write_content
    Uses only real tools that provide external capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration
        self.max_research_iterations = config.get("writer.max_research_iterations", 3)
        self.knowledge_coverage_threshold = config.get(
            "writer.knowledge_coverage_threshold", 0.7
        )
        self.max_research_queries_per_iteration = config.get(
            "writer.max_queries_per_iteration", 6
        )

        # Initialize real tools
        self.toolkit = AgentToolkit(config)

        # Build sophisticated workflow
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

    def process(self, topic: str) -> Article:
        """Process topic through sophisticated planning-first workflow."""

        logger.info(f"Starting sophisticated writer workflow for: {topic}")

        # Initialize state
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
            self_validation=None,
            research_iterations=0,
            ready_to_write=False,
            metadata={
                "method": "sophisticated_writer",
                "workflow_version": "2.0",
                "tool_usage": [],
                "decisions": [],
            },
        )

        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            article = self._create_article_from_state(final_state)

            logger.info(
                f"Completed sophisticated writer workflow for '{topic}' "
                f"({final_state['research_iterations']} research iterations, "
                f"{len(final_state['search_results'])} sources)"
            )

            return article

        except Exception as e:
            logger.error(f"Writer workflow failed for '{topic}': {e}")
            return self._create_error_article(topic, str(e))

    def _plan_outline_node(self, state: WriterState) -> WriterState:
        """Create initial outline using LLM reasoning (no tool needed)."""

        logger.info(f"Planning initial outline for: {state['topic']}")

        # LLM creates initial structure - this is reasoning, not external capability
        planning_prompt = f"""
        Create a detailed outline for a comprehensive article about: {state['topic']}

        Consider:
        - What are the essential aspects readers need to understand?
        - What background information is required?
        - What are the key concepts, applications, or examples?
        - What current developments or future implications exist?
        - How should information be logically organized?

        Create an outline with:
        1. A clear, specific title
        2. 4-6 main sections that thoroughly cover the topic
        3. Logical flow from basic concepts to advanced topics

        Format as:
        Title: [Specific Title]

        1. [Section 1 - Usually introduction/background]
        2. [Section 2 - Core concepts]
        3. [Section 3 - Applications/examples]
        4. [Section 4 - Current state/developments]
        5. [Section 5 - Future/implications]
        6. [Section 6 - Conclusion (if needed)]
        """

        outline_response = self.api_client.call_api(planning_prompt)
        initial_outline = self._parse_outline_response(outline_response, state["topic"])

        # Generate targeted research queries based on outline
        research_queries = self._generate_research_queries_from_outline(
            initial_outline, state["topic"]
        )

        state["initial_outline"] = initial_outline
        state["research_queries"] = research_queries
        state["metadata"]["decisions"].append(
            f"Created initial outline with {len(initial_outline.headings)} sections"
        )
        state["metadata"]["decisions"].append(
            f"Generated {len(research_queries)} targeted research queries"
        )

        logger.info(
            f"Created outline with {len(initial_outline.headings)} sections, "
            f"generated {len(research_queries)} research queries"
        )

        return state

    def _targeted_research_node(self, state: WriterState) -> WriterState:
        """Conduct targeted research using real search tools."""

        state["research_iterations"] += 1
        logger.info(
            f"Starting research iteration {state['research_iterations']} "
            f"with {len(state['research_queries'])} queries"
        )

        # Limit queries per iteration for efficiency
        current_queries = state["research_queries"][
            : self.max_research_queries_per_iteration
        ]

        new_results = []
        for query in current_queries:
            try:
                # Use real tool for external information retrieval
                search_result = self.toolkit.search_for_content(
                    query=query, purpose="writing"
                )

                if search_result["results"]:
                    new_results.extend(search_result["results"])
                    state["metadata"]["tool_usage"].append(
                        f"search_and_retrieve: '{query}' → {len(search_result['results'])} results"
                    )

            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # Add new results to existing ones
        state["search_results"].extend(new_results)

        # Use real tool to organize knowledge computationally
        if state["search_results"]:
            try:
                organized = self.toolkit.organize_for_writing(
                    topic=state["topic"], search_results=state["search_results"]
                )

                state["organized_knowledge"] = organized
                state["metadata"]["tool_usage"].append(
                    f"organize_knowledge: {len(state['search_results'])} results → "
                    f"{len(organized['categories'])} categories"
                )

            except Exception as e:
                logger.warning(f"Knowledge organization failed: {e}")

        logger.info(
            f"Research iteration {state['research_iterations']} completed: "
            f"{len(new_results)} new results, {len(state['search_results'])} total"
        )

        return state

    def _refine_outline_node(self, state: WriterState) -> WriterState:
        """Refine outline based on research findings using LLM reasoning."""

        logger.info("Refining outline based on research findings")

        if not state["organized_knowledge"]:
            # No knowledge to work with, keep original outline
            state["refined_outline"] = state["initial_outline"]
            state["ready_to_write"] = True
            return state

        # Analyze knowledge coverage for each outline section
        knowledge_coverage = self._analyze_knowledge_coverage(
            state["initial_outline"], state["organized_knowledge"]
        )

        # LLM refines outline based on research findings
        refinement_prompt = f"""
        Refine this outline for "{state['topic']}" based on research findings:

        Current Outline:
        {self._format_outline_for_prompt(state['initial_outline'])}

        Available Knowledge Categories:
        {state['organized_knowledge']['category_summary']}

        Knowledge Coverage Analysis:
        {knowledge_coverage}

        Instructions:
        1. Keep well-supported sections with sufficient information
        2. Modify or merge sections with insufficient coverage
        3. Add new sections for important topics discovered in research
        4. Ensure logical flow and comprehensive coverage
        5. Aim for 4-6 main sections maximum

        Provide the refined outline in the same format as the original.
        """

        refined_response = self.api_client.call_api(refinement_prompt)
        refined_outline = self._parse_outline_response(refined_response, state["topic"])

        # Identify knowledge gaps for potential additional research
        knowledge_gaps = self._identify_knowledge_gaps(
            refined_outline, state["organized_knowledge"]
        )

        state["refined_outline"] = refined_outline
        state["knowledge_gaps"] = knowledge_gaps

        state["metadata"]["decisions"].append(
            f"Refined outline: {len(state['initial_outline'].headings)} → "
            f"{len(refined_outline.headings)} sections"
        )

        if knowledge_gaps:
            state["metadata"]["decisions"].append(
                f"Identified {len(knowledge_gaps)} knowledge gaps"
            )

        logger.info(
            f"Outline refined: {len(state['initial_outline'].headings)} → "
            f"{len(refined_outline.headings)} sections, "
            f"{len(knowledge_gaps)} gaps identified"
        )

        return state

    def _write_content_node(self, state: WriterState) -> WriterState:
        """Generate article content using LLM with organized knowledge."""

        logger.info("Generating article content")

        if not state["refined_outline"]:
            # Fallback to initial outline
            working_outline = state["initial_outline"]
        else:
            working_outline = state["refined_outline"]

        if not working_outline:
            state["article_content"] = (
                f"# {state['topic']}\n\nUnable to create outline for article."
            )
            return state

        # Initialize article with title
        article_parts = [f"# {working_outline.title}"]

        # Generate each section using organized knowledge
        for section_heading in working_outline.headings:
            try:
                section_content = self._generate_section_content(
                    section_heading, state["topic"], state["organized_knowledge"]
                )

                article_parts.append(f"## {section_heading}")
                article_parts.append(section_content)

            except Exception as e:
                logger.warning(f"Failed to generate section '{section_heading}': {e}")
                article_parts.append(f"## {section_heading}")
                article_parts.append("Content generation failed for this section.")

        # Combine all parts
        full_content = "\n\n".join(article_parts)

        # Self-validate content using real NLP tool
        try:
            self_validation = self.toolkit.extract_content_claims(
                content=full_content, focus_types=["factual", "statistical"]
            )

            state["self_validation"] = self_validation
            state["metadata"]["tool_usage"].append(
                f"extract_claims: {self_validation['claims_found']} claims extracted for validation"
            )

        except Exception as e:
            logger.warning(f"Self-validation failed: {e}")

        state["article_content"] = full_content
        state["metadata"]["decisions"].append(
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
        if state["research_iterations"] >= self.max_research_iterations:
            logger.info("Max research iterations reached, proceeding to writing")
            state["ready_to_write"] = True
            return "ready_to_write"

        # Check knowledge coverage
        if state["organized_knowledge"]:
            coverage_score = state["organized_knowledge"].get("coverage_score", 0)

            if coverage_score >= self.knowledge_coverage_threshold:
                logger.info(
                    f"Knowledge coverage sufficient ({coverage_score:.2f}), proceeding to writing"
                )
                state["ready_to_write"] = True
                return "ready_to_write"

        # Check if we have significant knowledge gaps
        if len(state["knowledge_gaps"]) > 2:
            # Generate new research queries for gaps
            gap_queries = [
                f"detailed information about {gap} in context of {state['topic']}"
                for gap in state["knowledge_gaps"][:3]  # Limit new queries
            ]

            state["research_queries"] = gap_queries
            logger.info(
                f"Knowledge gaps found, conducting additional research for {len(gap_queries)} topics"
            )
            return "research_more"

        # Default: proceed to writing
        logger.info("Research appears sufficient, proceeding to writing")
        state["ready_to_write"] = True
        return "ready_to_write"

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

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

            # Specific information queries
            if "introduction" in heading.lower() or "background" in heading.lower():
                queries.append(f"overview of {topic}")
                queries.append(f"history of {topic}")
            elif "concept" in heading.lower() or "definition" in heading.lower():
                queries.append(f"definition {topic}")
                queries.append(f"key principles {topic}")
            elif "application" in heading.lower() or "example" in heading.lower():
                queries.append(f"applications of {topic}")
                queries.append(f"examples {topic}")
            elif "current" in heading.lower() or "development" in heading.lower():
                queries.append(f"recent developments {topic}")
                queries.append(f"current state {topic}")
            elif "future" in heading.lower() or "implication" in heading.lower():
                queries.append(f"future of {topic}")
                queries.append(f"implications {topic}")

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
        self, section_heading: str, topic: str, organized_knowledge: Dict
    ) -> str:
        """Generate content for a specific section using organized knowledge."""

        # Find relevant information for this section
        relevant_info = self._find_relevant_info_for_section(
            section_heading, organized_knowledge
        )

        # LLM generates content (this is text generation, not external capability)
        if relevant_info:
            content_prompt = f"""
            Write a comprehensive section titled "{section_heading}" for an article about "{topic}".

            Use this relevant information from research:
            {relevant_info}

            Requirements:
            - Write 2-3 well-structured paragraphs
            - Ground your content in the provided research information
            - Include specific details and examples from the sources
            - Maintain an informative, engaging tone
            - Ensure smooth transitions and logical flow
            - Avoid repetition of information from other sections
            """
        else:
            content_prompt = f"""
            Write a comprehensive section titled "{section_heading}" for an article about "{topic}".

            Requirements:
            - Write 2-3 well-structured paragraphs
            - Draw from your knowledge of {topic}
            - Provide clear explanations and relevant examples
            - Maintain an informative, engaging tone
            - Ensure logical flow and organization
            """

        return self.api_client.call_api(content_prompt)

    def _find_relevant_info_for_section(
        self, section_heading: str, organized_knowledge: Dict
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
        final_outline = state["refined_outline"] or state["initial_outline"]

        # Extract sections from content
        sections = {}
        if final_outline and state["article_content"]:
            import re

            section_pattern = r"## (.+?)\n\n(.*?)(?=\n## |\Z)"
            matches = re.findall(section_pattern, state["article_content"], re.DOTALL)
            for section_title, section_content in matches:
                sections[section_title.strip()] = section_content.strip()

        # Build comprehensive metadata
        metadata = state["metadata"].copy()
        metadata.update(
            {
                "research_iterations": state["research_iterations"],
                "total_search_results": len(state["search_results"]),
                "knowledge_coverage": (
                    state["organized_knowledge"].get("coverage_score", 0)
                    if state["organized_knowledge"]
                    else 0
                ),
                "knowledge_gaps": len(state["knowledge_gaps"]),
                "self_validation_claims": (
                    state["self_validation"].get("claims_found", 0)
                    if state["self_validation"]
                    else 0
                ),
                "word_count": len(state["article_content"].split()),
                "sections_generated": len(sections),
            }
        )

        return Article(
            title=final_outline.title if final_outline else state["topic"],
            content=state["article_content"],
            outline=final_outline,
            sections=sections,
            metadata=metadata,
        )

    def _create_error_article(self, topic: str, error_message: str) -> Article:
        """Create error article when workflow fails."""
        return Article(
            title=topic,
            content=f"# {topic}\n\nArticle generation failed: {error_message}",
            outline=Outline(title=topic, headings=["Error"], subheadings={"Error": []}),
            sections={"Error": error_message},
            metadata={
                "method": "sophisticated_writer",
                "error": error_message,
                "word_count": 0,
            },
        )
