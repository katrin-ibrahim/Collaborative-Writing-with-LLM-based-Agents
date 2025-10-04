# src/collaborative/agents/reviewer_agent.py
"""
LangGraph-based ReviewerAgent with structured workflow and automatic tool calling.
"""

from __future__ import annotations

import time

import json
import logging
import re
from langgraph.graph import END, StateGraph
from typing import Any, Dict, List, Optional, Tuple

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    agentic_review_prompt,
    enhanced_feedback_prompt,
    fact_checking_prompt,
    reviewer_search_strategy_prompt,
)
from src.collaborative.memory.memory import MemoryState
from src.collaborative.theory_of_mind import AgentRole
from src.collaborative.tools.reviewer_toolkit import ReviewerToolkit
from src.config.config_context import ConfigContext
from src.utils.data import Article, ArticleMetrics, FactCheckResult

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Sophisticated Reviewer agent with LangGraph workflow and automatic tool calling."""

    def __init__(self, llm=None):
        super().__init__()

        self.collaboration_config = ConfigContext.get_collaboration_config()
        self.retrieval_config = ConfigContext.get_retrieval_config()

        self.llm = llm or ConfigContext.get_tool_chat_client("writing")

        self.toolkit = ReviewerToolkit(self.retrieval_config)
        self.tools = self.toolkit.get_available_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}

        self.metrics_tool = self.tool_map.get("get_article_metrics")
        self.fact_check_tool = self.tool_map.get("verify_claims_with_research")

        self.max_claims_to_check = getattr(
            self.collaboration_config, "reviewer.max_claims_to_check", 10
        )

        self.workflow = self._build_workflow()

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(MemoryState)

        workflow.add_node("prepare", self._prepare_article_node)
        workflow.add_node("evaluate_feedback", self._evaluate_feedback_node)
        workflow.add_node("metrics_and_claims", self._metrics_and_claims_node)
        workflow.add_node("tool_planning", self._tool_planning_node)
        workflow.add_node("fact_check", self._fact_check_node)
        workflow.add_node("synthesize", self._synthesize_node)
        workflow.add_node("finish", self._finish_node)

        workflow.set_entry_point("prepare")
        workflow.add_edge("prepare", "evaluate_feedback")
        workflow.add_edge("evaluate_feedback", "metrics_and_claims")
        workflow.add_edge("metrics_and_claims", "tool_planning")
        workflow.add_edge("tool_planning", "fact_check")
        workflow.add_edge("fact_check", "synthesize")
        workflow.add_edge("synthesize", "finish")
        workflow.add_edge("finish", END)

        return workflow.compile()

    def process(self) -> None:
        shared_memory = ConfigContext.get_memory_instance()
        if not shared_memory:
            raise RuntimeError(
                "SharedMemory not available in ConfigContext. Initialize it before ReviewerAgent.process()."
            )

        shared_memory.state.setdefault("metadata", {}).setdefault(
            "review_decisions", []
        )

        try:
            result_state = self.workflow.invoke(shared_memory.state)
            shared_memory.state.update(result_state)
        except Exception as e:
            logger.error(f"Reviewer workflow failed: {e}")
            # Provide minimal feedback so collaboration can continue
            try:
                shared_memory.store_structured_feedback(
                    feedback_text="Basic review: Article appears complete but could not perform detailed analysis due to technical issues.",
                    iteration=shared_memory.get_current_iteration(),
                    target_section=None,
                )
            except Exception as feedback_error:
                logger.error(f"Failed to store fallback feedback: {feedback_error}")
                # Store feedback directly in memory state as last resort
                feedback_history = shared_memory.state.setdefault(
                    "feedback_history", []
                )
                feedback_history.append(
                    {
                        "feedback_text": "Technical review completed with limitations.",
                        "iteration": shared_memory.get_current_iteration(),
                        "suggestions": [],
                    }
                )

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------
    def _prepare_article_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_memory = ConfigContext.get_memory_instance()
        iteration = shared_memory.get_current_iteration()

        # Debug: Check what's in shared memory
        logger.info(
            f"🔍 DEBUG: SharedMemory state keys: {list(shared_memory.state.keys())}"
        )
        logger.info(
            f"🔍 DEBUG: topic = '{shared_memory.state.get('topic', 'MISSING')}'"
        )
        logger.info(
            f"🔍 DEBUG: article_content = '{shared_memory.state.get('article_content', 'MISSING')[:100]}...' (len={len(shared_memory.state.get('article_content', ''))})"
        )
        logger.info(f"🔍 DEBUG: iteration = {iteration}")

        sections = shared_memory.get_sections_from_iteration(iteration)
        logger.info(
            f"🔍 DEBUG: sections from iteration {iteration} = {list(sections.keys()) if sections else 'NONE'}"
        )

        article = Article(
            title=shared_memory.state.get("topic", ""),
            content=shared_memory.state.get("article_content", ""),
            sections=sections,
            metadata=shared_memory.state.get("metadata", {}),
        )

        logger.info(
            f"Reviewer preparing article '{article.title}' (iteration {iteration})"
        )
        logger.info(
            f"🔍 DEBUG: Created article - title='{article.title}', content_len={len(article.content)}, article_obj={type(article)}"
        )
        return {"_current_article": article}

    def _evaluate_feedback_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_memory = ConfigContext.get_memory_instance()
        if shared_memory.state.get("iteration", 0) > 0:
            self._evaluate_writer_feedback_claims(shared_memory.state)
        return {}

    def _metrics_and_claims_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        article: Article = state.get("_current_article")
        logger.info(
            f"🔍 DEBUG: Metrics node - article = {article}, type = {type(article)}"
        )
        logger.info(f"🔍 DEBUG: Metrics node - state keys: {list(state.keys())}")
        if article:
            logger.info(
                f"🔍 DEBUG: Article has content: {len(article.content) if hasattr(article, 'content') else 'NO CONTENT ATTR'}"
            )
        metrics = self._get_article_metrics(article)
        state["_metrics"] = metrics

        shared_memory = ConfigContext.get_memory_instance()
        tom_module = getattr(shared_memory, "tom_module", None)
        if tom_module and tom_module.enabled:
            prediction = tom_module.predict_agent_response(
                predictor=AgentRole.REVIEWER,
                target=AgentRole.WRITER,
                context={
                    "article_metrics": metrics.to_dict(),
                    "previous_feedback_severity": self._assess_feedback_severity(
                        shared_memory
                    ),
                    "current_feedback": shared_memory.state.get("metadata", {}).get(
                        "current_feedback", []
                    ),
                },
            )
            shared_memory.state.setdefault("metadata", {})[
                "reviewer_writer_prediction"
            ] = {
                "predicted_action": prediction.predicted_action,
                "confidence": prediction.confidence,
                "reasoning": prediction.reasoning,
            }

        potential_claims = self._extract_claims_for_fact_checking(article)
        return {"_metrics": metrics, "_potential_claims": potential_claims}

    def _tool_planning_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        article: Article = state.get("_current_article")
        metrics: ArticleMetrics = state.get("_metrics")
        potential_claims: List[str] = state.get("_potential_claims", [])

        metrics_summary = json.dumps(metrics.to_dict(), indent=2)
        claims_summary = "\n".join(
            f"- {claim}" for claim in potential_claims[: self.max_claims_to_check]
        )

        prompt = (
            "You are deciding which reviewer tools to use before writing feedback.\n"
            "Respond EXACTLY in this format:\n"
            "USE_VERIFY: YES or NO\n"
            "REQUEST_SEARCH: YES or NO\n"
            "SEARCH_QUERIES: query1 | query2 (if REQUEST_SEARCH is YES, else NONE)\n"
            "INSPECT_CHUNKS: chunk_id, ... (list up to 5 chunk IDs to inspect with get_chunks_by_ids, or NONE)\n"
            "NOTES: short explanation\n\n"
            f"Article: {article.title}\n"
            f"Metrics:\n{metrics_summary}\n\n"
            f"Claims to verify (top {self.max_claims_to_check}):\n{claims_summary or 'None'}\n"
            "Prefer using existing writer chunks; only request new searches if absolutely necessary."
        )

        try:
            response = self.api_client.call_api(prompt=prompt, system_prompt=None)
            if not isinstance(response, str):
                response = str(response) if response is not None else ""
        except Exception as e:
            logger.error(f"API client call failed in tool planning: {e}")
            response = "USE_VERIFY: NO\nREQUEST_SEARCH: NO\nSEARCH_QUERIES: NONE\nINSPECT_CHUNKS: NONE\nNOTES: Technical issue with LLM"
        logger.debug(
            "LLM reviewer-plan len=%d preview=%s",
            len(response or ""),
            (response or "")[:200].replace("\n", " ⏎ "),
        )
        selected_tools, search_flag, queries, inspect_chunks, notes = (
            self._parse_reviewer_plan_response(response)
        )

        logger.info(
            f"Reviewer planner → tools: {selected_tools}, search: {queries}, inspect_chunks: {inspect_chunks}"
        )

        if search_flag and queries and "search_and_retrieve" in self.tool_map:
            for query in queries[:3]:
                try:
                    result = self.tool_map["search_and_retrieve"].invoke(query)
                    if result.get("success"):
                        logger.info(
                            f"Reviewer planner search '{query}' stored {result.get('total_chunks', 0)} chunks"
                        )
                except Exception as exc:
                    logger.warning(
                        f"Reviewer planner search failed for '{query}': {exc}"
                    )

        state["_review_notes"] = state.get("_review_notes", []) + [notes]
        state["_inspect_chunks"] = inspect_chunks

        if "verify_claims_with_research" not in selected_tools:
            selected_tools.append("verify_claims_with_research")

        return {
            "_review_notes": state.get("_review_notes", []) + [notes],
            "_inspect_chunks": inspect_chunks,
            "_selected_tools": selected_tools,
        }

    def _fact_check_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        claims = state.get("_potential_claims", [])
        fact_check_results = self._fact_check_claims(claims)
        return {"_fact_check_results": fact_check_results}

    def _synthesize_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_memory = ConfigContext.get_memory_instance()
        article: Article = state.get("_current_article")
        metrics: ArticleMetrics = state.get("_metrics")
        fact_check_results: List[FactCheckResult] = state.get("_fact_check_results", [])
        potential_claims = state.get("_potential_claims", [])

        selected_tools = state.get("_selected_tools", [])
        inspect_chunks = state.get("_inspect_chunks", [])
        supplementary_context = self._execute_selected_tools(
            selected_tools, article, inspect_chunks
        )

        qualitative_feedback = self._generate_feedback(
            article,
            metrics,
            potential_claims,
            fact_check_results,
            supplementary_context,
        )

        issues, recommendations = self._parse_qualitative_feedback(qualitative_feedback)
        overall_score = self._calculate_overall_score(metrics, fact_check_results)

        shared_memory.store_structured_feedback(
            feedback_text=qualitative_feedback,
            iteration=shared_memory.get_current_iteration(),
            target_section=None,
        )

        feedback_metadata = {
            "overall_score": overall_score,
            "issues": issues,
            "recommendations": recommendations,
            "metrics": metrics.to_dict(),
            "fact_check_results": [result.to_dict() for result in fact_check_results],
            "tools_used": selected_tools,
            "timestamp": time.time(),
        }

        shared_memory.state.setdefault("metadata", {}).setdefault(
            "review_results", []
        ).append(feedback_metadata)

        return {
            "_overall_score": overall_score,
            "_qualitative_feedback": qualitative_feedback,
        }

    def _finish_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        shared_memory = ConfigContext.get_memory_instance()
        notes = state.get("_review_notes", [])
        if notes:
            shared_memory.state.setdefault("metadata", {}).setdefault(
                "review_decisions", []
            ).extend(notes)
        return {}

    # ------------------------------------------------------------------
    # Helper logic copied/adapted from original reviewer agent
    # ------------------------------------------------------------------
    def _get_article_metrics(self, article: Article) -> ArticleMetrics:
        if not self.metrics_tool:
            raise RuntimeError("get_article_metrics tool not available")
        if article is None:
            raise RuntimeError("Article is None - cannot get metrics")
        if not hasattr(article, "content"):
            raise RuntimeError(
                f"Article object has no content attribute: {type(article)}"
            )

        # Pass both content and title as required by the tool
        result = self.metrics_tool.invoke(
            {"content": article.content, "title": article.title}
        )
        logger.info(f"🔍 DEBUG: Metrics tool result: {result}")

        if isinstance(result, dict):
            # Ensure we have all required fields for ArticleMetrics
            metrics_data = {
                "title": article.title,
                "word_count": result.get("word_count", 0),
                "character_count": len(article.content),
                "heading_count": result.get("heading_count", 0),
                "headings": [],  # TODO: Extract headings properly
                "paragraph_count": result.get("paragraph_count", 0),
                "analysis_success": True,
            }
            return ArticleMetrics(**metrics_data)
        if isinstance(result, ArticleMetrics):
            return result
        raise RuntimeError("Unexpected metrics tool result")

    def _fact_check_claims(self, claims: List[str]) -> List[FactCheckResult]:
        claims = [claim.strip() for claim in claims if claim.strip()]
        claims = claims[: self.max_claims_to_check]
        if not claims:
            return []

        # Skip LLM reasoning step and go directly to tool-based verification
        logger.info(f"Fact-checking {len(claims)} claims using verification tool")

        if not self.fact_check_tool:
            return []
        try:
            verification = self.fact_check_tool.invoke(claims)
        except Exception as exc:
            logger.warning(f"Fact-check tool failed: {exc}")
            return []

        results: List[FactCheckResult] = []
        for entry in verification.get("verifications", []):
            facts = FactCheckResult(
                claim=entry.get("claim", ""),
                sources_found=entry.get("relevant_chunks_found", 0),
                search_successful=True,
                verified=entry.get("verified", False),
                search_results=entry.get("chunk_contents", []),
                error=entry.get("error"),
            )
            results.append(facts)
        return results

    def _generate_feedback(
        self,
        article: Article,
        metrics: ArticleMetrics,
        potential_claims: List[str],
        fact_check_results: List[FactCheckResult],
        supplementary_context: str,
    ) -> str:
        prompt = enhanced_feedback_prompt(
            article=article,
            metrics=metrics,
            potential_claims=potential_claims,
            fact_check_results=fact_check_results,
            supplementary_context=supplementary_context,
        )
        text = self.api_client.call_api(prompt=prompt, system_prompt=None)
        return text if isinstance(text, str) else str(text)

    def _execute_selected_tools(
        self, selected_tools: List[str], article: Article, inspect_chunks: List[str]
    ) -> str:
        context_parts: List[str] = []
        tool_map = {tool.name: tool for tool in self.tools}

        for tool_name in selected_tools:
            if tool_name == "search_and_retrieve" and tool_name in tool_map:
                prompt = reviewer_search_strategy_prompt(
                    article.title, article.sections
                )
                decision = self.api_client.call_api(prompt=prompt, system_prompt=None)
                if not isinstance(decision, str):
                    decision = str(decision)
                logger.debug(
                    "LLM reviewer-search-strategy len=%d preview=%s",
                    len(decision or ""),
                    (decision or "")[:200].replace("\n", " ⏎ "),
                )
                queries = [q.strip() for q in decision.split("\n") if q.strip()]
                for query in queries[
                    : self.collaboration_config.reviewer_max_search_results
                ]:
                    try:
                        result = tool_map[tool_name].invoke(query)
                        if result.get("success"):
                            context_parts.append(
                                f"Search '{query}' returned {result.get('total_chunks', 0)} chunks"
                            )
                    except Exception as exc:
                        logger.warning(f"Reviewer search failed for '{query}': {exc}")

        if inspect_chunks and "get_chunks_by_ids" in tool_map:
            try:
                result = tool_map["get_chunks_by_ids"].invoke(
                    ids=",".join(inspect_chunks)
                )
                if result.get("success"):
                    context_parts.append(
                        f"Inspected chunks: {', '.join(inspect_chunks)}"
                    )
            except Exception as exc:
                logger.warning(f"Failed to inspect chunks {inspect_chunks}: {exc}")

        return "\n".join(context_parts)

    def _generate_agentic_feedback(
        self,
        article: Article,
        metrics: ArticleMetrics,
        fact_check_results: List[FactCheckResult],
    ) -> str:
        prompt = agentic_review_prompt(article, metrics, fact_check_results)
        try:
            text = self.api_client.call_api(prompt=prompt, system_prompt=None)
            return (
                text
                if isinstance(text, str)
                else (
                    str(text)
                    if text is not None
                    else "Review completed with technical limitations."
                )
            )
        except Exception as e:
            logger.error(f"API client call failed in feedback generation: {e}")
            return "Review completed with technical limitations due to LLM issues."

    def _parse_qualitative_feedback(self, feedback: str) -> Tuple[List[str], List[str]]:
        issues = self._extract_section(feedback, "## CURRENT SHORTCOMINGS")
        recommendations = self._extract_section(
            feedback, "## ACTIONABLE RECOMMENDATIONS"
        )
        return issues, recommendations

    def _calculate_overall_score(
        self, metrics: ArticleMetrics, fact_check_results: List[FactCheckResult]
    ) -> float:
        base_score = 0.5
        base_score += min(metrics.word_count / 1500, 1.0) * 0.2
        base_score += min(metrics.heading_count / 8, 1.0) * 0.2
        if fact_check_results:
            verified = sum(1 for result in fact_check_results if result.verified)
            base_score += min(verified / max(len(fact_check_results), 1), 1.0) * 0.3
        return round(min(base_score, 1.0), 3)

    def _extract_claims_for_fact_checking(self, article: Article) -> List[str]:
        # Use the fact_checking_prompt with correct parameters
        prompt = fact_checking_prompt(article.content, article.title)
        response = self.api_client.call_api(prompt=prompt, system_prompt=None)
        if not isinstance(response, str):
            response = str(response)
        logger.debug(
            "LLM claim-extract len=%d preview=%s",
            len(response or ""),
            (response or "")[:200].replace("\n", " ⏎ "),
        )

        # Extract claims from the XML response
        claims = []
        # Simple extraction - look for <text> tags
        import re

        claim_matches = re.findall(r"<text>(.*?)</text>", response, re.DOTALL)
        for claim in claim_matches:
            cleaned = claim.strip()
            if cleaned:
                claims.append(cleaned)

        # Fallback: if no XML found, try line-by-line extraction
        if not claims:
            for line in response.splitlines():
                cleaned = line.strip("- *\t ")
                if cleaned and len(cleaned) > 10:  # Skip very short lines
                    claims.append(cleaned)

        return claims[: self.max_claims_to_check]

    # Removed helper: call engine directly at call sites and use logger.debug

    def _normalize_output(self, result: Any) -> Tuple[str, List[Dict[str, Any]]]:
        tool_calls: List[Dict[str, Any]] = []
        if result is None:
            logger.warning("Result is None in _normalize_output")
            return "", tool_calls
        if isinstance(result, dict):
            if result.get("tool_calls"):
                tool_calls = result["tool_calls"]
            return str(result.get("content", "")), tool_calls
        if hasattr(result, "content"):
            content = getattr(result, "content", "") or ""
            additional_kwargs = getattr(result, "additional_kwargs", {})
            tool_calls = additional_kwargs.get("tool_calls", [])
            return str(content), tool_calls
        return str(result), tool_calls

    def _parse_reviewer_plan_response(
        self, response: str
    ) -> Tuple[List[str], bool, List[str], List[str], str]:
        selected_tools: List[str] = []
        search_flag = False
        queries: List[str] = []
        inspect_chunks: List[str] = []
        notes = ""

        for line in response.splitlines():
            clean = line.strip()
            if clean.upper().startswith("USE_VERIFY:"):
                value = clean[len("USE_VERIFY:") :].strip().upper()
                if value.startswith("Y"):
                    selected_tools.append("verify_claims_with_research")
            elif clean.upper().startswith("REQUEST_SEARCH:"):
                value = clean[len("REQUEST_SEARCH:") :].strip().upper()
                search_flag = value.startswith("Y")
                if search_flag and "search_and_retrieve" not in selected_tools:
                    selected_tools.append("search_and_retrieve")
            elif clean.upper().startswith("SEARCH_QUERIES:"):
                value = clean[len("SEARCH_QUERIES:") :].strip()
                if value.upper() != "NONE":
                    queries = [q.strip() for q in value.split("|") if q.strip()]
            elif clean.upper().startswith("INSPECT_CHUNKS:"):
                value = clean[len("INSPECT_CHUNKS:") :].strip()
                if value.upper() != "NONE":
                    inspect_chunks = [
                        cid.strip() for cid in value.split(",") if cid.strip()
                    ]
                    if inspect_chunks and "get_chunks_by_ids" not in selected_tools:
                        selected_tools.append("get_chunks_by_ids")
            elif clean.upper().startswith("NOTES:"):
                notes = clean[len("NOTES:") :].strip()

        if "verify_claims_with_research" not in selected_tools:
            selected_tools.insert(0, "verify_claims_with_research")
        return (
            selected_tools,
            search_flag,
            queries,
            inspect_chunks,
            notes or "Planner response unavailable",
        )

    def _run_tool_conversation(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_names: List[str],
        max_turns: int = 4,
    ) -> Optional[Dict[str, Any]]:
        tools = [self.tool_map[name] for name in tool_names if name in self.tool_map]
        tool_lookup = {tool.name: tool for tool in tools}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        model = self.llm.bind_tools(tools) if tools else self.llm
        final_text = ""

        for _ in range(max_turns):
            try:
                result = model.invoke(messages)
                if result is None:
                    logger.error("LLM returned None result - model invocation failed")
                    return {
                        "content": "Error: Model invocation failed",
                        "tool_calls": [],
                    }
                content, tool_calls = self._normalize_output(result)
                assistant_msg = {"role": "assistant", "content": content}
            except Exception as e:
                logger.error(f"Error during model invocation: {e}")
                return {
                    "content": f"Error during model invocation: {e}",
                    "tool_calls": [],
                }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "name": call.get("name")
                        or call.get("function", {}).get("name"),
                        "args": (
                            call.get("arguments")
                            or call.get("args")
                            or json.loads(
                                call.get("function", {}).get("arguments", "{}")
                            )
                            if isinstance(
                                call.get("function", {}).get("arguments"), str
                            )
                            else call.get("function", {}).get("arguments", {})
                        ),
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_msg)

            if assistant_msg.get("tool_calls"):
                for call in assistant_msg["tool_calls"]:
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
            try:
                start = cleaned.index("{")
                end = cleaned.rindex("}") + 1
                return json.loads(cleaned[start:end])
            except Exception:
                return None

    def _extract_section(self, text: str, header: str) -> List[str]:
        pattern = re.escape(header) + r"\s*(.*?)(?=\n##|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        section = match.group(1)
        items: List[str] = []
        for line in section.split("\n"):
            cleaned = line.strip()
            if re.match(r"^[-•*]\s+", cleaned):
                cleaned = re.sub(r"^[-•*]\s+", "", cleaned)
                if len(cleaned) > 3:
                    items.append(cleaned)
            elif cleaned:
                items.append(cleaned)
        return items[:10]

    def _calculate_feedback_severity(self, feedback_items: List[Dict[str, Any]]) -> str:
        if not feedback_items:
            return "none"
        if len(feedback_items) >= 5:
            return "high"
        if len(feedback_items) >= 3:
            return "medium"
        return "low"

    def _assess_feedback_severity(self, shared_memory) -> str:
        feedback = shared_memory.state.get("structured_feedback", [])
        recent = [
            item
            for item in feedback
            if item.get("iteration") == shared_memory.state.get("iteration", 0) - 1
        ]
        return self._calculate_feedback_severity(recent)

    def _evaluate_writer_feedback_claims(self, state: MemoryState):
        feedback_list = state.get("structured_feedback", [])
        if not feedback_list:
            return
        memory = ConfigContext.get_memory_instance()
        for feedback in feedback_list:
            status = feedback.get("status")
            if status == "claimed_addressed":
                content = state.get("article_content", "")
                if self._verify_feedback_addressed(feedback, content):
                    memory.mark_feedback_verified_by_reviewer(
                        feedback["id"], "verified_addressed"
                    )
                else:
                    memory.mark_feedback_verified_by_reviewer(
                        feedback["id"], "not_addressed"
                    )
            elif status == "contested":
                self._handle_contested_feedback(
                    feedback, state.get("article_content", "")
                )

    def _handle_contested_feedback(
        self, feedback_item: Dict[str, Any], current_article: str
    ):
        prompt = reviewer_search_strategy_prompt(
            feedback_item.get("feedback") or feedback_item.get("text", ""),
            current_article,
        )
        decision = self.api_client.call_api(prompt=prompt)
        resolution = "maintain_reviewer_position"
        reasoning = decision.strip()
        memory = ConfigContext.get_memory_instance()
        memory.resolve_contested_feedback(
            feedback_item.get("id"),
            resolution,
            "reviewer",
            reasoning,
        )

    def _verify_feedback_addressed(self, feedback_item, current_article: str) -> bool:
        feedback_text = str(
            feedback_item.get("feedback") or feedback_item.get("text", "")
        ).lower()
        article_lower = current_article.lower()
        quoted_terms = re.findall(r'"([^"]+)"', feedback_text)
        keywords = [
            term.lower().strip() for term in quoted_terms if len(term.strip()) > 2
        ]
        for keyword in keywords:
            if keyword not in article_lower:
                return False
        return True
