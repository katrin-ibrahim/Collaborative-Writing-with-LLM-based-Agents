# src/collaborative/agents/writer_agent.py
"""
LangGraph-based WriterAgent with sophisticated workflow and automatic tool calling.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from langgraph.graph import END, StateGraph
from typing import Any, Dict, List, Optional, Tuple

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    context_decision_prompt,
    planning_prompt,
    search_query_generation_prompt,
    section_content_prompt_with_research,
    section_content_prompt_without_research,
)
from src.collaborative.memory.memory import MemoryState
from src.collaborative.tools.writer_toolkit import WriterToolkit
from src.config.config_context import ConfigContext
from src.utils.data import Article, Outline

logger = logging.getLogger(__name__)


@dataclass
class ToolPlannerResult:
    selected_chunks: List[str]
    notes: str


class WriterAgent(BaseAgent):
    """Sophisticated Writer agent with LangGraph workflow and automatic tool calling."""

    def __init__(self, llm=None):
        super().__init__()

        self.collaboration_config = ConfigContext.get_collaboration_config()
        self.retrieval_config = ConfigContext.get_retrieval_config()

        # Choose ChatModel adapter for LangGraph tool usage
        self.llm = llm or ConfigContext.get_tool_chat_client("writing")

        # Toolkit and tools
        self.toolkit = WriterToolkit(self.retrieval_config)
        self.tools = self.toolkit.get_available_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

        self.search_tool = self.tool_map.get("search_and_retrieve")
        self.feedback_tool = self.tool_map.get("get_feedback")

        # Config values
        self.num_queries = getattr(self.retrieval_config, "num_queries", 5)

        self.workflow = self._build_workflow()

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(MemoryState)

        workflow.add_node("search", self._search_node)
        workflow.add_node("outline", self._outline_node)
        workflow.add_node("decide_context", self._decide_context_node)
        workflow.add_node("tool_planning", self._tool_planning_node)
        workflow.add_node("write", self._write_node)
        workflow.add_node("evaluate_continue", self._evaluate_continue_node)

        workflow.set_entry_point("search")
        workflow.add_edge("search", "outline")
        workflow.add_edge("outline", "decide_context")
        workflow.add_edge("decide_context", "tool_planning")
        workflow.add_edge("tool_planning", "write")
        workflow.add_edge("write", "evaluate_continue")
        workflow.add_conditional_edges(
            "evaluate_continue",
            self._route_after_evaluation,
            {"tools": "tool_planning", "end": END},
        )

        return workflow.compile()

    def process(self) -> None:
        shared_memory = ConfigContext.get_memory_instance()
        if not shared_memory:
            raise RuntimeError(
                "SharedMemory not available in ConfigContext. Initialize it before running WriterAgent.process()."
            )

        # Ensure metadata scaffolding exists
        metadata = shared_memory.state.setdefault("metadata", {})
        metadata.setdefault("decisions", [])
        metadata.setdefault("workflow_version", "lg_hybrid_1.0")
        metadata.setdefault("method", "writer_agent_lg")

        result_state = self.workflow.invoke(shared_memory.state)
        shared_memory.state.update(result_state)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run initial research on iteration 0."""
        iteration = state.get("iteration", 0)
        if iteration > 0:
            logger.info("SEARCH_NODE: Skipping initial search (later iteration)")
            return {"metadata": state.get("metadata", {})}

        topic = state.get("topic", "")
        logger.info(
            f"SEARCH_NODE: Starting initial research for: {topic}, search_tool={self.search_tool is not None}"
        )

        direct_search_chunks = 0
        context = ""

        if self.search_tool:
            try:
                logger.info(f"SEARCH_NODE: Calling search_tool.invoke('{topic}')")
                direct_search = self.search_tool.invoke(topic)
                if direct_search.get("success"):
                    direct_search_chunks = direct_search.get("total_chunks", 0)
                    summaries = direct_search.get("chunk_summaries", [])
                    if summaries:
                        context = summaries[0].get("description", "")
                else:
                    logger.warning(
                        f"SEARCH_NODE: Direct search failed for '{topic}': {direct_search}"
                    )
            except Exception as exc:
                logger.warning(f"SEARCH_NODE: Direct search exception: {exc}")

        prompt = search_query_generation_prompt(topic, context, self.num_queries)
        queries_response = self.api_client.call_api(prompt=prompt)
        queries = self._parse_queries(queries_response)

        total_chunks_stored = direct_search_chunks
        for query in queries:
            if not self.search_tool:
                break
            try:
                logger.info(f"SEARCH_NODE: Calling search_tool.invoke('{query}')")
                search_result = self.search_tool.invoke(query)
                if search_result.get("success"):
                    chunks_found = search_result.get("total_chunks", 0)
                    total_chunks_stored += chunks_found
                    logger.info(
                        f"SEARCH_NODE: Query '{query}' stored {chunks_found} chunks"
                    )
                else:
                    logger.warning(
                        f"SEARCH_NODE: Query '{query}' failed: {search_result}"
                    )
            except Exception as exc:
                logger.warning(f"SEARCH_NODE: Search exception for '{query}': {exc}")

        decisions = state.setdefault("metadata", {}).setdefault("decisions", [])
        decisions.append(
            f"Initial research complete: {len(queries)} targeted queries, {total_chunks_stored} total chunks"
        )

        return {"metadata": state["metadata"]}

    def _outline_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create outline using stored research."""
        topic = state.get("topic", "")
        logger.info(f"Creating outline for: {topic}")

        # Select best single chunk by stored similarity to ground the outline
        memory = ConfigContext.get_memory_instance()
        best_chunk = None
        best_score = None
        for ch in memory.get_stored_chunks():
            score = ch.metadata.get("relevance_score")
            if score is None:
                # fallback to inverse rank (lower is better)
                rank = ch.metadata.get("relevance_rank", 9999)
                score_val = -float(rank)
            else:
                score_val = float(score)
            if best_score is None or score_val > best_score:
                best_score = score_val
                best_chunk = ch

        excerpt = (
            (best_chunk.content[:1200] + "...")
            if best_chunk and len(best_chunk.content) > 1200
            else (best_chunk.content if best_chunk else "")
        )

        prompt = (
            planning_prompt(topic)
            + "\n\nUse the following excerpt to ground the outline.\n"
            + "EXCERPT:\n"
            + (excerpt or "")
            + "\n\nCreate an entity-rich outline: prefer real team names, venues, dates, and match facts present in the excerpt."
        )
        system = (
            "Headings only. After the title line, output six lines starting with '## '. "
            "No prose paragraphs. Base headings on the excerpt — do not invent facts."
        )
        outline_response = self.api_client.call_api(
            prompt=prompt,
            system_prompt=system,
            temperature=0.2,
            max_tokens=180,
            stop=["\n\n"],
        )
        if not isinstance(outline_response, str):
            outline_response = str(outline_response)
        logger.debug(
            "LLM outline len=%d preview=%s",
            len(outline_response or ""),
            (outline_response or "")[:200].replace("\n", " ⏎ "),
        )
        # Parse headings in a permissive way (agentic): collect all '## ' lines
        outline = self._parse_outline(outline_response, topic)
        state["initial_outline"] = outline

        logger.info(
            "Outline ready with %d sections: %s",
            len(outline.headings),
            outline.headings,
        )
        state["metadata"]["decisions"].append(
            f"Outline ready with {len(outline.headings)} sections"
        )
        return {"initial_outline": outline, "metadata": state["metadata"]}

    def _decide_context_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_memory = ConfigContext.get_memory_instance()
        logger.info("Deciding whether more context is needed")

        search_summaries = shared_memory.get_search_summaries()

        headings = []
        if state.get("initial_outline"):
            headings = state["initial_outline"].headings

        prompt = context_decision_prompt(
            state.get("topic", ""),
            search_summaries,
            ", ".join(headings),
            len(search_summaries),
        )
        decision_response = self.api_client.call_api(prompt=prompt)
        decision = decision_response.strip().lower()

        logger.info("Context decision raw response: %s", decision_response.strip())
        state["metadata"]["context_decision"] = decision
        state["metadata"]["decisions"].append(f"Context decision: {decision}")
        return {"metadata": state["metadata"]}

    def _tool_planning_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_memory = ConfigContext.get_memory_instance()
        metadata = state.setdefault("metadata", {})
        metadata.setdefault("tool_attempts", 0)
        metadata["tool_attempts"] += 1

        logger.info(f"Tool planning iteration (attempt {metadata['tool_attempts']})")

        if state.get("iteration", 0) > 0:
            self._gather_pending_feedback(state, shared_memory)

        chunk_summaries = self._collect_chunk_summaries(shared_memory)

        planner_result = self._run_tool_planner(state, chunk_summaries)
        if planner_result and planner_result.selected_chunks:
            metadata["selected_chunks"] = planner_result.selected_chunks
            notes = planner_result.notes or ""
        else:
            fallback = list(chunk_summaries.keys())[:5]
            metadata["selected_chunks"] = fallback
            notes = "Planner fallback: using top chunk summaries"

        metadata.setdefault("decisions", []).append(
            f"Tool planner notes: {notes[:120]}"
        )
        return {"metadata": state["metadata"]}

    def _write_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from src.config.config_context import ConfigContext

        shared_memory = ConfigContext.get_memory_instance()
        logger.info("Generating article content")

        outline: Outline = state.get("initial_outline")
        if not outline:
            fallback = f"# {state.get('topic', 'Article')}\n\nUnable to create outline."
            self._persist_article(shared_memory, state, outline, fallback, {})
            return {"article_content": fallback, "metadata": state.get("metadata", {})}

        iteration = state.get("iteration", 0)
        sections_to_revise = set(self._get_sections_to_revise(shared_memory))

        article_parts = [f"# {outline.title}"]
        section_contents: Dict[str, str] = {}

        logger.info("Outline headings entering write: %s", outline.headings)

        for heading in outline.headings:
            try:
                logger.info(
                    "Preparing section '%s' (regenerate=%s)",
                    heading,
                    iteration == 0 or heading in sections_to_revise,
                )
                regenerate = iteration == 0 or heading in sections_to_revise
                if regenerate:
                    section_text = self._generate_section_content(
                        heading, state, shared_memory
                    )
                    logger.info(f"Generated section '{heading}'")
                else:
                    section_text = self._get_existing_section_content(
                        heading, shared_memory
                    )
                    logger.info(f"Reused section '{heading}'")
            except Exception as exc:
                logger.warning(f"Section '{heading}' generation failed: {exc}")
                section_text = "Content generation failed for this section."

            article_parts.append(f"## {heading}")
            article_parts.append(section_text)
            section_contents[heading] = section_text

        full_article = "\n\n".join(article_parts)
        state["article_content"] = full_article

        if len(full_article.strip()) < 400:
            logger.warning(
                "Article draft too short (%d chars). Building fallback summary.",
                len(full_article.strip()),
            )
            summary_chunks = state.get("metadata", {}).get("selected_chunks", [])
            fallback_sections: List[str] = []
            for chunk_id in summary_chunks[:4]:
                chunk = shared_memory.get_chunk_by_id(chunk_id)
                if chunk and chunk.content:
                    fallback_sections.append(chunk.content[:500])
            fallback_content = "\n\n".join(fallback_sections) or "Summary unavailable."
            title = outline.title if outline else state.get("topic", "Article")
            full_article = f"# {title}\n\n{fallback_content}"
            state["article_content"] = full_article
            if outline and outline.headings:
                section_contents = {outline.headings[0]: fallback_content}
            else:
                section_contents = {"Summary": fallback_content}

        self._persist_article(
            shared_memory, state, outline, full_article, section_contents
        )

        logger.info(
            "Generated article length: %d chars across %d sections",
            len(full_article),
            len(section_contents),
        )
        logger.info("Article preview (first 400 chars): %s", full_article[:400])

        if iteration > 0 and sections_to_revise:
            self._mark_section_feedback_applied(shared_memory, list(sections_to_revise))

        decision_msg = (
            f"Revised {len(sections_to_revise)} sections"
            if sections_to_revise
            else "Initial article draft created"
        )
        state["metadata"]["decisions"].append(decision_msg)

        return {
            "article_content": full_article,
            "article_sections_by_iteration": {str(iteration): section_contents},
            "metadata": state["metadata"],
        }

    def _evaluate_continue_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"metadata": state.get("metadata", {})}

    def _route_after_evaluation(self, state: Dict[str, Any]) -> str:
        metadata = state.setdefault("metadata", {})
        tool_attempts = metadata.get("tool_attempts", 0)
        max_attempts = 3
        content_length = len(state.get("article_content", ""))

        if tool_attempts >= max_attempts:
            logger.warning("Max tool attempts reached, ending workflow")
            return "end"

        if content_length < 500:
            logger.info("Article still short, looping back to tools")
            return "tools"

        return "end"

    # ------------------------------------------------------------------
    # Tool planner (hybrid auto-plan logic)
    # ------------------------------------------------------------------
    def _run_tool_planner(
        self, state: Dict[str, Any], chunk_summaries: Dict[str, str]
    ) -> Optional[ToolPlannerResult]:
        topic = state.get("topic", "")
        iteration = state.get("iteration", 0)
        outline = state.get("initial_outline")
        outline_text = ", ".join(outline.headings) if outline else ""
        feedback_items = state.get("metadata", {}).get("current_feedback", [])

        chunk_lines = [
            f"{cid}: {summary}" for cid, summary in list(chunk_summaries.items())[:10]
        ]
        feedback_summary = "\n".join(
            f"- {item.get('id')}: {item.get('feedback') or item.get('text', '')}"
            for item in feedback_items[:5]
        )

        prompt = (
            "You are planning what the writer should do before drafting the next version.\n"
            "Follow this exact response template (no extra text):\n"
            "SELECT_CHUNKS: chunk_id, ... (list up to 5 chunk IDs or NONE)\n"
            "REQUEST_SEARCH: YES or NO\n"
            "SEARCH_QUERIES: query1 | query2 (only if REQUEST_SEARCH is YES, otherwise NONE)\n"
            "NOTES: short explanation\n\n"
            f"Topic: {topic}\n"
            f"Iteration: {iteration}\n"
            f"Outline sections: {outline_text}\n"
            f"Pending feedback items: {len(feedback_items)}\n"
            f"Top feedback (if any):\n{feedback_summary or 'None'}\n"
            f"Available chunk summaries (max 10):\n{chr(10).join(chunk_lines) or 'None'}\n"
            "Remember: reuse existing chunks whenever possible. Only request a search if critical information is missing."
        )

        response = self.api_client.call_api(prompt=prompt)
        selected_chunks, search_flag, queries, notes = self._parse_writer_plan_response(
            response, chunk_summaries
        )

        logger.info(
            f"Writer planner → chunks: {selected_chunks}, request_search: {search_flag}, queries: {queries}"
        )

        if search_flag and queries and self.search_tool:
            for query in queries[:3]:
                try:
                    result = self.search_tool.invoke(query)
                    if result.get("success"):
                        logger.info(
                            f"Planner search '{query}' stored {result.get('total_chunks', 0)} chunks"
                        )
                except Exception as exc:
                    logger.warning(f"Planner search failed for '{query}': {exc}")

        return ToolPlannerResult(selected_chunks=selected_chunks, notes=notes)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _parse_writer_plan_response(
        self, response: str, chunk_summaries: Dict[str, str]
    ) -> Tuple[List[str], bool, List[str], str]:
        selected: List[str] = []
        request_search = False
        queries: List[str] = []
        notes = ""

        for line in response.splitlines():
            clean = line.strip()
            if clean.upper().startswith("SELECT_CHUNKS:"):
                value = clean[len("SELECT_CHUNKS:") :].strip()
                if value.upper() != "NONE":
                    selected = [
                        cid.strip()
                        for cid in value.split(",")
                        if cid.strip() in chunk_summaries
                    ]
            elif clean.upper().startswith("REQUEST_SEARCH:"):
                value = clean[len("REQUEST_SEARCH:") :].strip().upper()
                request_search = value.startswith("Y")
            elif clean.upper().startswith("SEARCH_QUERIES:"):
                value = clean[len("SEARCH_QUERIES:") :].strip()
                if value.upper() != "NONE":
                    queries = [q.strip() for q in value.split("|") if q.strip()]
            elif clean.upper().startswith("NOTES:"):
                notes = clean[len("NOTES:") :].strip()

        if not selected:
            selected = list(chunk_summaries.keys())[:5]
        return (
            selected,
            request_search,
            queries,
            notes or "Planner response unavailable",
        )

    def _normalize_llm_output(self, result: Any) -> Tuple[str, List[Dict[str, Any]]]:
        tool_calls: List[Dict[str, Any]] = []

        if result is None:
            return "", tool_calls

        if isinstance(result, dict):
            content = result.get("content") or ""
            if result.get("tool_calls"):
                tool_calls = [
                    {
                        "name": call.get("name"),
                        "args": call.get("arguments") or call.get("args") or {},
                    }
                    for call in result["tool_calls"]
                ]
            return str(content), tool_calls

        # LangChain AIMessage
        if hasattr(result, "content"):
            content = result.content or ""
            additional_kwargs = getattr(result, "additional_kwargs", {})
            llm_calls = additional_kwargs.get("tool_calls") or []
            for call in llm_calls:
                tool_calls.append(
                    {
                        "name": call.get("function", {}).get("name")
                        or call.get("name"),
                        "args": (
                            json.loads(call.get("function", {}).get("arguments", "{}"))
                            if isinstance(
                                call.get("function", {}).get("arguments"), str
                            )
                            else call.get("function", {}).get("arguments", {})
                        ),
                    }
                )
            return str(content), tool_calls

        return str(result), tool_calls

    def _run_tool_conversation(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_names: List[str],
        max_turns: int = 5,
    ) -> Optional[Dict[str, Any]]:
        tools = [self.tool_map[name] for name in tool_names if name in self.tool_map]
        tool_lookup = {tool.name: tool for tool in tools}

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        model = self.llm.bind_tools(tools) if tools else self.llm
        final_text = ""

        for _ in range(max_turns):
            result = model.invoke(messages)
            content, tool_calls = self._normalize_llm_output(result)
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if tool_calls:
                for call in tool_calls:
                    tool_name = call.get("name")
                    args = call.get("args", {})
                    tool_obj = tool_lookup.get(tool_name)
                    if not tool_obj:
                        tool_result = {
                            "success": False,
                            "error": f"Unknown tool '{tool_name}'",
                        }
                    else:
                        try:
                            tool_result = tool_obj.invoke(**args)
                        except Exception as exc:
                            tool_result = {
                                "success": False,
                                "error": str(exc),
                            }
                    messages.append(
                        {
                            "role": "tool",
                            "name": tool_name or "tool",
                            "content": json.dumps(tool_result),
                        }
                    )
                continue

            final_text = content
            break

        return {"messages": messages, "final": final_text}

    def _parse_planner_output(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        cleaned = text.strip()
        if cleaned.upper().startswith("FINAL:"):
            cleaned = cleaned[6:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON substring
            try:
                start = cleaned.index("{")
                end = cleaned.rindex("}") + 1
                return json.loads(cleaned[start:end])
            except Exception:
                return None

    def _gather_pending_feedback(self, state: Dict[str, Any], shared_memory):
        if not self.feedback_tool:
            return
        try:
            feedback_result = self.feedback_tool.invoke(only_pending=True)
        except Exception as exc:
            logger.warning(f"Failed to retrieve feedback via tool: {exc}")
            return

        if feedback_result.get("success"):
            feedback_items = feedback_result.get("feedback", [])
            metadata = state.setdefault("metadata", {})
            metadata["current_feedback"] = feedback_items
            metadata["feedback_count"] = len(feedback_items)

            sections = {
                item.get("target_section")
                for item in feedback_items
                if item.get("target_section")
            }
            metadata["sections_to_revise"] = list(sections)
            logger.info(
                f"Loaded {len(feedback_items)} pending feedback items for revision"
            )
        else:
            logger.info("No pending feedback retrieved via tool")

    def _collect_chunk_summaries(self, shared_memory) -> Dict[str, str]:
        summaries: Dict[str, str] = {}
        search_summaries = shared_memory.get_search_summaries()
        for result in search_summaries.values():
            # Ensure result is a dictionary before calling .get()
            if not isinstance(result, dict):
                logger.warning(
                    f"Skipping non-dict search result: {type(result)} = {result}"
                )
                continue

            chunk_summaries = result.get("chunk_summaries", [])
            if not isinstance(chunk_summaries, list):
                logger.warning(
                    f"chunk_summaries is not a list: {type(chunk_summaries)}"
                )
                continue

            for chunk_summary in chunk_summaries:
                if not isinstance(chunk_summary, dict):
                    logger.warning(
                        f"Skipping non-dict chunk_summary: {type(chunk_summary)}"
                    )
                    continue

                chunk_id = chunk_summary.get("chunk_id")
                if not chunk_id:
                    continue
                description = chunk_summary.get("description", "")
                source = chunk_summary.get("source", "")
                label = description
                if source:
                    label = f"{description} (Source: {source})"
                summaries[chunk_id] = label
        return summaries

    def _persist_article(
        self,
        shared_memory,
        state: Dict[str, Any],
        outline: Optional[Outline],
        content: str,
        sections: Dict[str, str],
    ) -> None:
        article_metadata = state.get("metadata", {}).copy()
        article = Article(
            title=outline.title if outline else state.get("topic", "Article"),
            content=content,
            outline=outline,
            sections=sections,
            metadata=article_metadata,
        )
        shared_memory.update_article_state(article)
        state["drafts_by_iteration"] = shared_memory.state["drafts_by_iteration"]
        state["article_sections_by_iteration"] = shared_memory.state[
            "article_sections_by_iteration"
        ]

    def _parse_queries(self, response: str) -> List[str]:
        queries: List[str] = []
        for line in response.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if cleaned.lower().startswith(("here are", "query:", "search:")):
                continue
            cleaned = re.sub(r"^\d+[.):]\s*", "", cleaned)
            cleaned = cleaned.strip("\"'")
            if len(cleaned) >= 3:
                queries.append(cleaned)
        return queries[: self.num_queries]

    def _parse_outline(self, response: str, topic: str) -> Outline:
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        # Title: first '# ' line or fallback
        title = topic
        for ln in lines:
            if ln.startswith("# "):
                title = ln.lstrip("# ").strip()
                break
        # Collect all '## ' headings in order (agentic, no strict cap)
        headings: List[str] = [
            ln.lstrip("# ").strip() for ln in lines if ln.startswith("## ")
        ]
        if not headings:
            # Fallback to a minimal outline if model failed
            headings = [
                "Overview",
                "Teams and Venue",
                "Route to Grand Final",
                "Key Match Moments",
                "Standout Players",
                "Result and Aftermath",
            ]
        subheadings: Dict[str, List[str]] = {h: [] for h in headings}
        return Outline(title=title, headings=headings, subheadings=subheadings)

    def _generate_section_content(
        self, heading: str, state: Dict[str, Any], shared_memory
    ) -> str:
        selected_chunks = state.get("metadata", {}).get("selected_chunks", [])
        chunk_contents: List[str] = []
        for chunk_id in selected_chunks:
            chunk = shared_memory.get_chunk_by_id(chunk_id)
            if chunk:
                chunk_contents.append(chunk.content)

        relevant_info = "\n\n".join(chunk_contents)
        feedback_text = self._get_section_feedback(heading, shared_memory)
        topic = state.get("topic", "")

        if relevant_info or feedback_text:
            if feedback_text:
                feedback_block = f"\n\nREVIEWER FEEDBACK TO ADDRESS:\n{feedback_text}"
                relevant_info = (
                    relevant_info + feedback_block
                    if relevant_info
                    else feedback_block.strip()
                )
            prompt = section_content_prompt_with_research(
                section_heading=heading,
                topic=topic,
                relevant_info=relevant_info,
            )
        else:
            prompt = section_content_prompt_without_research(
                section_heading=heading,
                topic=topic,
            )

        text = self.api_client.call_api(prompt=prompt)
        logger.info("Raw section response for '%s': %r", heading, text)
        logger.info(
            "Writer section output for '%s' (len=%d): %s",
            heading,
            len(text or ""),
            (text or "")[:200],
        )

        if not text or len(text.strip()) < 200:
            logger.warning(
                "Section '%s' output too short (%d chars). Using chunk fallback.",
                heading,
                len(text.strip()) if text else 0,
            )
            fallback = (
                "\n\n".join(chunk_contents[:2])
                or "Content unavailable from research chunks."
            )
            text = (
                f"{heading}\n\n"
                f"Summary based on available research about {topic}:\n\n"
                f"{fallback[:1000]}"
            )

        return text

    def _get_section_feedback(self, heading: str, shared_memory) -> str:
        feedback_items = shared_memory.state.get("metadata", {}).get(
            "current_feedback", []
        )
        section_feedback = [
            item.get("feedback") or item.get("text", "")
            for item in feedback_items
            if item.get("target_section") == heading
        ]
        return "\n".join(section_feedback)

    def _get_sections_to_revise(self, shared_memory) -> List[str]:
        sections = set()
        for feedback in shared_memory.state.get("structured_feedback", []):
            if feedback.get("status") == "pending" and feedback.get("target_section"):
                sections.add(feedback["target_section"])
        sections.update(
            shared_memory.state.get("metadata", {}).get("sections_to_revise", [])
        )
        return list(sections)

    def _get_existing_section_content(self, heading: str, shared_memory) -> str:
        state = shared_memory.state
        iteration = state.get("iteration", 0)
        if iteration > 0:
            prev = state.get("article_sections_by_iteration", {}).get(
                str(iteration - 1), {}
            )
            if heading in prev:
                return prev[heading]
        article_text = state.get("article_content", "")
        if article_text:
            pattern = rf"## {re.escape(heading)}\n\n(.*?)(?=\n## |\Z)"
            match = re.search(pattern, article_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        logger.warning(f"No existing content for section '{heading}', regenerating")
        return ""

    def _mark_section_feedback_applied(
        self, shared_memory, sections_revised: List[str]
    ):
        metadata = shared_memory.state.get("metadata", {})
        current_feedback = metadata.get("current_feedback", [])
        for feedback in current_feedback:
            target = feedback.get("target_section")
            fid = feedback.get("id")
            if target in sections_revised and fid:
                reasoning = f"Updated section '{target}' in iteration {shared_memory.state.get('iteration', 0)}"
                shared_memory.mark_feedback_claimed_by_writer(
                    fid,
                    claim_status="claimed_addressed",
                    reasoning=reasoning,
                )
