# src/agents/agentic_writer.py
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from agents.base_agent import BaseAgent
from agents.tools.agent_toolkit import AgentToolkit
from knowledge.knowledge_base import KnowledgeBase
from utils.data_models import Article, Outline, SearchResult


class WriterState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    topic: str
    current_phase: str
    search_results: List[SearchResult]
    knowledge_base: Optional[KnowledgeBase]
    outline: Optional[Outline]
    article_content: str
    sections_completed: List[str]
    needs_more_info: bool
    confidence_score: float
    metadata: Dict[str, Any]


class WriterAgent(BaseAgent):
    """
    Configurable agentic writer using LangGraph for autonomous decision-making.

    The agent makes decisions within configured constraints:
    - use_external_knowledge: Whether agent can search for information
    - use_knowledge_organization: Whether agent can organize information
    - knowledge_depth: Level of organization sophistication

    This allows the same agentic framework to power different baseline workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Configuration constraints that affect agent behavior
        self.use_external_knowledge = config.get("writer.use_external_knowledge", True)
        self.use_knowledge_organization = config.get(
            "writer.use_knowledge_organization", True
        )
        self.knowledge_depth = config.get("writer.knowledge_depth", "basic")
        self.max_search_iterations = config.get("writer.max_search_iterations", 3)

        self.toolkit = AgentToolkit(config)
        self.workflow = self._build_workflow_graph()

        self.logger.info(
            f"AgenticWriter configured: external_knowledge={self.use_external_knowledge}, "
            f"knowledge_org={self.use_knowledge_organization}, depth={self.knowledge_depth}"
        )

    def _build_workflow_graph(self) -> StateGraph:
        """Build LangGraph workflow for autonomous writing decisions."""
        workflow = StateGraph(WriterState)

        workflow.add_node("plan_approach", self._plan_approach)
        workflow.add_node("gather_information", self._gather_information)
        workflow.add_node("organize_knowledge", self._organize_knowledge)
        workflow.add_node("plan_outline", self._plan_outline)
        workflow.add_node("write_content", self._write_content)
        workflow.add_node("evaluate_progress", self._evaluate_progress)

        workflow.set_entry_point("plan_approach")

        workflow.add_conditional_edges(
            "plan_approach",
            self._decide_next_action,
            {
                "research": "gather_information",
                "organize": "organize_knowledge",
                "outline": "plan_outline",
                "write": "write_content",
            },
        )

        workflow.add_conditional_edges(
            "gather_information",
            self._decide_after_research,
            {
                "research_more": "gather_information",
                "organize": "organize_knowledge",
                "outline": "plan_outline",
            },
        )

        workflow.add_conditional_edges(
            "organize_knowledge",
            self._decide_after_organization,
            {
                "research_more": "gather_information",
                "outline": "plan_outline",
                "write": "write_content",
            },
        )

        workflow.add_edge("plan_outline", "write_content")

        workflow.add_conditional_edges(
            "write_content",
            self._decide_after_writing,
            {
                "continue_writing": "write_content",
                "research_more": "gather_information",
                "evaluate": "evaluate_progress",
                "complete": END,
            },
        )

        workflow.add_conditional_edges(
            "evaluate_progress",
            self._decide_after_evaluation,
            {
                "research_more": "gather_information",
                "write_more": "write_content",
                "reorganize": "organize_knowledge",
                "complete": END,
            },
        )

        return workflow.compile()

    def process(self, topic: str) -> Article:
        """Process topic through autonomous decision-making workflow."""
        initial_state = WriterState(
            messages=[
                HumanMessage(content=f"Write a comprehensive article about: {topic}")
            ],
            topic=topic,
            current_phase="planning",
            search_results=[],
            knowledge_base=None,
            outline=None,
            article_content="",
            sections_completed=[],
            needs_more_info=True,
            confidence_score=0.0,
            metadata={"method": "agentic_writer", "decisions": []},
        )

        final_state = self.workflow.invoke(initial_state)
        return self._state_to_article(final_state)

    def _plan_approach(self, state: WriterState) -> WriterState:
        """Agent decides initial approach within configuration constraints."""
        if not self.use_external_knowledge:
            # Internal knowledge only - agent knows it can't search
            planning_prompt = f"""
            I need to write a comprehensive article about "{state['topic']}" using only my internal knowledge.

            Analyze this topic and decide my approach:
            1. What's my confidence level (0.0-1.0) in my internal knowledge of this topic?
            2. Should I start with an outline or write directly?
            3. How should I structure my approach for best results?

            Since I cannot search for external information, focus on:
            - How well I know this topic internally
            - What structure would work best
            - Whether to outline first or write directly

            Respond with just one word: OUTLINE or WRITE
            """

            response = self.call_api(planning_prompt).strip().upper()
            state["confidence_score"] = (
                0.7  # Reasonable confidence for internal knowledge
            )
            state["needs_more_info"] = False  # Can't search anyway

        else:
            # Full capabilities - agent can decide to research
            planning_prompt = f"""
            I need to write a comprehensive article about "{state['topic']}".

            Analyze this topic and decide my initial approach:
            1. Do I have enough internal knowledge to start writing immediately?
            2. Should I research this topic first to gather current information?
            3. Is this a complex topic that needs structured research and organization?
            4. What's my confidence level (0.0-1.0) in my current knowledge?

            Consider factors like:
            - How technical/specialized the topic is
            - Whether current/recent information is important
            - How broad or narrow the topic scope is

            Respond with just one word: RESEARCH, ORGANIZE, OUTLINE, or WRITE
            """

            response = self.call_api(planning_prompt).strip().upper()

            # Set confidence based on decision
            if "WRITE" in response:
                state["confidence_score"] = 0.8
                state["needs_more_info"] = False
            elif "OUTLINE" in response:
                state["confidence_score"] = 0.6
                state["needs_more_info"] = False
            else:
                state["confidence_score"] = 0.3
                state["needs_more_info"] = True

        state["metadata"]["decisions"].append(
            f"Initial approach: {response} (config: external_knowledge={self.use_external_knowledge})"
        )
        state["current_phase"] = "planned"

        return state

    def _gather_information(self, state: WriterState) -> WriterState:
        """Agent decides what information to gather within configuration constraints."""
        if not self.use_external_knowledge:
            # This shouldn't be called in internal-only mode, but handle gracefully
            state["metadata"]["decisions"].append(
                "Skipped information gathering (internal knowledge only)"
            )
            return state

        # Check search iteration limit
        current_searches = len(
            [d for d in state["metadata"]["decisions"] if "Searched for:" in d]
        )
        if current_searches >= self.max_search_iterations:
            state["metadata"]["decisions"].append(
                f"Reached max search iterations ({self.max_search_iterations})"
            )
            state["needs_more_info"] = False
            return state

        # Analyze what information is needed
        info_planning_prompt = f"""
        I'm researching "{state['topic']}" for a comprehensive article.

        Current research status:
        - Existing search results: {len(state['search_results'])}
        - Current phase: {state['current_phase']}
        - Search iterations so far: {current_searches}/{self.max_search_iterations}

        What specific information do I need? Consider:
        1. What are the key aspects I should research?
        2. What search queries would get me the most valuable information?
        3. Should I focus on Wikipedia, web sources, or both?

        Provide 2-3 specific search queries that would give me comprehensive coverage.
        Format: one query per line, no numbers or bullets.
        """

        response = self.call_api(info_planning_prompt)
        search_queries = [
            q.strip()
            for q in response.split("\n")
            if q.strip() and not q.strip().startswith(("1.", "2.", "3.", "-", "*"))
        ]

        # Execute searches based on agent's decision
        new_results = []
        for query in search_queries[:3]:  # Agent can decide up to 3 queries
            # Agent decides source priority
            if any(
                word in query.lower()
                for word in ["recent", "current", "latest", "news"]
            ):
                results = self.toolkit.search.search_web(query, max_results=4)
            else:
                results = self.toolkit.search.search_all_sources(
                    query, wiki_results=3, web_results=2
                )
            new_results.extend(results)

        state["search_results"].extend(new_results)
        state["metadata"]["decisions"].append(f"Searched for: {search_queries}")
        state["current_phase"] = "researched"

        return state

    def _organize_knowledge(self, state: WriterState) -> WriterState:
        """Agent decides how to organize information within configuration constraints."""
        if not state["search_results"]:
            return state

        if not self.use_knowledge_organization:
            # Simple organization - just store results without sophisticated structure
            organizer = self.toolkit.knowledge.create_organizer(state["topic"])
            organizer.add_search_results(state["search_results"])
            state["knowledge_base"] = organizer
            state["metadata"]["decisions"].append(
                "Used simple organization (knowledge organization disabled)"
            )
            state["current_phase"] = "organized"
            return state

        # Agent decides organization strategy based on knowledge_depth
        if self.knowledge_depth == "basic":
            # Simple categorization
            categories = [
                {"name": "Overview", "description": "Introduction and basic concepts"},
                {"name": "Details", "description": "Specific information and examples"},
                {
                    "name": "Current State",
                    "description": "Recent developments and status",
                },
            ]
        else:
            # Sophisticated categorization - let agent decide
            organization_prompt = f"""
            I have {len(state['search_results'])} search results about "{state['topic']}".

            How should I organize this information for writing? Consider:
            1. What are the main conceptual categories for this topic?
            2. How should I group the information logically?
            3. What organization would make the most sense for readers?

            Provide 4-5 category names that would organize this information well.
            Format as JSON:
            {{
                "categories": [
                    {{"name": "Category 1", "description": "What this covers"}},
                    {{"name": "Category 2", "description": "What this covers"}}
                ]
            }}
            """

            response = self.call_api(organization_prompt)

            # Parse agent's organization decision
            try:
                import json
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    org_plan = json.loads(json_match.group())
                    categories = org_plan.get("categories", [])
                else:
                    raise ValueError("No JSON found")
            except:
                # Fallback if parsing fails
                categories = [
                    {
                        "name": "Fundamentals",
                        "description": "Core concepts and principles",
                    },
                    {
                        "name": "Technical Aspects",
                        "description": "Technical details and mechanisms",
                    },
                    {
                        "name": "Applications",
                        "description": "Real-world uses and examples",
                    },
                    {
                        "name": "Current Developments",
                        "description": "Recent trends and future directions",
                    },
                ]

        # Create and organize using agent's decisions (or defaults)
        organizer = self.toolkit.knowledge.create_organizer(state["topic"])
        self.toolkit.knowledge.organize_for_writing(
            organizer, categories, state["search_results"]
        )

        state["knowledge_base"] = organizer
        state["metadata"]["decisions"].append(
            f"Organized into categories: {[c['name'] for c in categories]} (depth: {self.knowledge_depth})"
        )
        state["current_phase"] = "organized"

        return state

    def _plan_outline(self, state: WriterState) -> WriterState:
        """Agent decides outline structure based on available information."""
        # Prepare context from organized knowledge
        context = ""
        if state["knowledge_base"]:
            summary = state["knowledge_base"].get_content_summary()
            context = f"Available categories: {[cat['name'] for cat in summary['categories']]}"
        elif state["search_results"]:
            context = self.toolkit.content.create_context_from_results(
                state["search_results"][:5], max_length=1000
            )

        # Agent generates outline using available information
        outline = self.toolkit.content.generate_outline(state["topic"], context)
        state["outline"] = outline

        state["metadata"]["decisions"].append(
            f"Created outline with {len(outline.headings)} sections"
        )
        state["current_phase"] = "outlined"

        return state

    def _write_content(self, state: WriterState) -> WriterState:
        """Agent decides what section to write and how to write it."""
        if not state["outline"]:
            return state

        # Agent decides which section to write next
        remaining_sections = [
            h for h in state["outline"].headings if h not in state["sections_completed"]
        ]
        if not remaining_sections:
            return state

        # Agent can choose section priority
        section_selection_prompt = f"""
        I need to write the next section for my article about "{state['topic']}".

        Remaining sections: {remaining_sections}
        Already completed: {state['sections_completed']}

        Which section should I write next? Consider:
        1. Logical flow and reader experience
        2. Foundation needed for other sections
        3. Information availability

        Respond with just the section name from the list above.
        """

        response = self.call_api(section_selection_prompt).strip()

        # Find matching section (fuzzy match)
        next_section = remaining_sections[0]  # Default
        for section in remaining_sections:
            if (
                section.lower() in response.lower()
                or response.lower() in section.lower()
            ):
                next_section = section
                break

        # Agent decides what context to use for this section
        context = ""
        if state["knowledge_base"]:
            context = state["knowledge_base"].get_content_by_category(next_section)
            if not context:
                # Find most relevant content
                relevant_results = self.toolkit.knowledge.find_relevant_content(
                    state["knowledge_base"], next_section, max_results=3
                )
                context = self.toolkit.content.create_context_from_results(
                    relevant_results
                )
        elif state["search_results"]:
            relevant_results = state["search_results"][:3]  # Simple fallback
            context = self.toolkit.content.create_context_from_results(relevant_results)

        # Generate section content
        section_content = self.toolkit.content.generate_section_content(
            next_section, context, state["topic"], self.api_client
        )

        # Add to article
        if not state["article_content"]:
            state["article_content"] = f"# {state['outline'].title}\n\n"

        state[
            "article_content"
        ] += f"## {next_section}\n\n{section_content.strip()}\n\n"
        state["sections_completed"].append(next_section)

        state["metadata"]["decisions"].append(f"Wrote section: {next_section}")
        state["current_phase"] = "writing"

        return state

    def _evaluate_progress(self, state: WriterState) -> WriterState:
        """Agent evaluates current progress and decides next steps."""
        if not state["knowledge_base"]:
            available_info = self.toolkit.content.create_context_from_results(
                state["search_results"]
            )
        else:
            summary = state["knowledge_base"].get_content_summary()
            available_info = (
                f"Organized info with {summary['total_search_results']} sources"
            )

        # Agent evaluates completeness
        assessment = self.toolkit.evaluation.assess_content_gaps(
            state["topic"], state["sections_completed"], available_info, self.api_client
        )

        state["confidence_score"] = assessment.get("overall_completeness", 0.5)
        state["needs_more_info"] = not assessment.get("information_sufficient", True)

        state["metadata"]["decisions"].append(
            f"Evaluation: completeness={state['confidence_score']:.2f}"
        )
        state["current_phase"] = "evaluated"

        return state

    # Decision functions for conditional edges
    def _decide_next_action(self, state: WriterState) -> str:
        """Agent's initial decision about what to do within configuration constraints."""
        if not self.use_external_knowledge:
            # Internal knowledge only - skip research and organization
            if not state["outline"]:
                return "outline"
            else:
                return "write"

        # Full capabilities mode
        if state["confidence_score"] < 0.5:
            return "research"
        elif (
            state["search_results"]
            and not state["knowledge_base"]
            and self.use_knowledge_organization
        ):
            return "organize"
        elif not state["outline"]:
            return "outline"
        else:
            return "write"

    def _decide_after_research(self, state: WriterState) -> str:
        """Agent decides what to do after gathering information."""
        if len(state["search_results"]) < 5:
            return "research_more"
        elif len(state["search_results"]) > 10 and self.use_knowledge_organization:
            return "organize"
        else:
            return "outline"

    def _decide_after_organization(self, state: WriterState) -> str:
        """Agent decides what to do after organizing knowledge."""
        if (
            state["confidence_score"] < 0.6
            and len(state["search_results"]) < 15
            and self.use_external_knowledge
        ):
            return "research_more"
        elif not state["outline"]:
            return "outline"
        else:
            return "write"

    def _decide_after_writing(self, state: WriterState) -> str:
        """Agent decides what to do after writing content."""
        if not state["outline"]:
            return "complete"

        remaining = [
            h for h in state["outline"].headings if h not in state["sections_completed"]
        ]

        if not remaining:
            return "complete"
        elif (
            state["confidence_score"] < 0.5
            and self.use_external_knowledge
            and len(state["search_results"]) < 20
        ):
            return "research_more"
        elif len(remaining) > 2 and len(state["sections_completed"]) % 3 == 0:
            return "evaluate"
        else:
            return "continue_writing"

    def _decide_after_evaluation(self, state: WriterState) -> str:
        """Agent decides what to do after evaluating progress."""
        remaining = [
            h for h in state["outline"].headings if h not in state["sections_completed"]
        ]

        if not remaining:
            return "complete"
        elif state["needs_more_info"] and self.use_external_knowledge:
            return "research_more"
        elif (
            state["confidence_score"] < 0.4
            and self.use_knowledge_organization
            and state["search_results"]
        ):
            return "reorganize"
        else:
            return "write_more"

    def _state_to_article(self, state: WriterState) -> Article:
        """Convert final state to Article object."""
        sections = {}
        if state["outline"] and state["sections_completed"]:
            # Extract sections from article content
            import re

            section_pattern = r"## (.+?)\n\n(.*?)(?=\n## |\Z)"
            matches = re.findall(section_pattern, state["article_content"], re.DOTALL)
            for section_title, section_content in matches:
                sections[section_title.strip()] = section_content.strip()

        metadata = state["metadata"].copy()
        metadata.update(
            {
                "word_count": len(state["article_content"].split()),
                "sections_completed": len(state["sections_completed"]),
                "total_searches": len(state["search_results"]),
                "confidence_score": state["confidence_score"],
                "final_phase": state["current_phase"],
            }
        )

        if state["knowledge_base"]:
            metadata["sources"] = state["knowledge_base"].get_all_sources()
            metadata["organization_categories"] = state[
                "knowledge_base"
            ].get_categories()

        return Article(
            title=state["outline"].title if state["outline"] else state["topic"],
            content=state["article_content"],
            outline=state["outline"],
            sections=sections,
            metadata=metadata,
        )
