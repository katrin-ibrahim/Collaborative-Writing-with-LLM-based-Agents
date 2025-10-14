# src/collaborative/agents/writer_v2.py
import logging
from typing import Any, Dict, List, Optional

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import (
    outline_prompt,
    search_query_generation_prompt,
    select_section_chunks_prompt,
    write_section_content_prompt,
)
from src.collaborative.tools.tool_definitions import (
    get_chunks_by_ids,
    get_feedback,
    search_and_retrieve,
)
from src.config.config_context import ConfigContext
from src.utils.data import Article, Outline, ResearchChunk

logger = logging.getLogger(__name__)


class WriterV2(BaseAgent):
    """
    Simplified Writer agent with hybrid Python/LLM architecture.

    Workflow:
    - Iteration 0: direct_search(topic) → generate_queries(top_chunk) →
                   search_queries → create_outline(top_chunk) → write_all_sections(chunks)
    - Iteration 1+: get_pending_feedback → identify_target_sections →
                     revise_sections(feedback + chunks) → mark_feedback_status
    """

    def __init__(self):
        super().__init__()
        self.shared_memory = ConfigContext.get_memory_instance()
        if not self.shared_memory:
            raise RuntimeError("SharedMemory instance not found in ConfigContext")

        self.tom_module = self.shared_memory.tom_module

        # Minimal tool set - Python makes all decisions
        self.search_tool = search_and_retrieve
        self.chunks_tool = get_chunks_by_ids
        self.feedback_tool = get_feedback

        self.retrieval_config = ConfigContext.get_retrieval_config()
        self.num_queries = getattr(self.retrieval_config, "num_queries", 5)
        self.num_chunks = getattr(self.retrieval_config, "max_content_pieces", 5)

    def process(self) -> None:
        """Main entry point with clear iteration routing."""
        iteration = self.shared_memory.get_current_iteration()
        logger.info(f"WriterV2 processing iteration {iteration}")

        if iteration == 0:
            logger.info("Executing initial research and drafting workflow")
            self._execute_iteration_0()
        else:
            logger.info("Executing revision workflow based on feedback")
            self._execute_iteration_plus()

    def _execute_iteration_0(self):
        """
        Deterministic workflow: direct_search(topic) → generate_queries(top_chunk) →
        search_queries → create_outline(top_chunk) → write_all_sections(chunks)
        """
        topic = self.shared_memory.state.get("topic")
        if not topic:
            raise RuntimeError("No topic found in SharedMemory state")

        logger.info(f"WriterV2 starting iteration 0 for topic: {topic}")

        # Step 1: Direct search (Python decision)
        logger.info("Step 1: Executing direct search")
        self._search(topic, rm_type="supabase_faiss")

        # Step 2: Get top chunk for context (Python logic)
        logger.info("Step 2: Retrieving top chunk for context")
        top_chunk = self._get_best_chunk()

        # Step 3: Generate secondary queries (LLM reasoning) - using fast model
        logger.info("Step 3: Generating secondary queries with LLM")
        queries = self._generate_queries(topic, top_chunk)

        # Step 4: Execute secondary searches (concurrent execution)
        logger.info(f"Step 4: Executing {len(queries)} secondary searches concurrently")
        self._search_concurrent(queries)

        # Step 5: Create outline (LLM reasoning with top chunk) - using outline model
        logger.info("Step 5: Creating outline with LLM")
        chunk_summaries = self.shared_memory.get_search_summaries()
        max_content_pieces = self.retrieval_config.max_content_pieces
        formatted_chunk_summaries = (
            "["
            + "; ".join(
                f"{q}: {{"
                + ", ".join(
                    f"{c.get('description','').strip()}"  # Use full smart descriptions from generate_description
                    for c in v.get("chunk_summaries", [])[:max_content_pieces]
                )
                + "}}"
                for q, v in chunk_summaries.items()
            )
            + "]"
        )
        outline = self._outline(topic, top_chunk, formatted_chunk_summaries)

        # Step 6: Write all sections (Python orchestration + LLM content)
        logger.info(f"Step 6: Writing {len(outline.headings)} sections")
        self._write_all_sections(outline)

        logger.info("WriterV2 iteration 0 completed successfully")

    def _execute_iteration_plus(self):
        """
        Deterministic workflow: get_pending_feedback → identify_target_sections →
        revise_sections(feedback + chunks) → mark_feedback_status
        """
        iteration = self.shared_memory.get_current_iteration()
        logger.info(f"WriterV2 starting iteration {iteration}")

        # Step 1: Get pending feedback (Python decision)
        logger.info("Step 1: Retrieving pending feedback")
        feedback_result = self._get_pending_feedback()

        # Step 2: Identify sections to revise (Python logic)
        logger.info("Step 2: Identifying sections to revise")
        sections_to_revise = self._extract_target_sections(feedback_result)
        logger.info(
            f"Found {len(sections_to_revise)} sections to revise: {sections_to_revise}"
        )

        # Step 3: For each section, revise with feedback (Python loop + LLM content)
        for section in sections_to_revise:
            logger.info(f"Step 3: Revising section '{section}'")
            self._revise_section(section)

        # Step 4: Mark feedback as addressed using existing memory methods
        logger.info("Step 4: Marking feedback as addressed")
        self._mark_feedback_addressed(sections_to_revise)

        logger.info("WriterV2 iteration completed successfully")

    @staticmethod
    def _search(query: str, rm_type: str = None) -> Dict[str, Any]:
        """Consolidated search method for both direct and secondary searches."""
        # Call the tool with proper LangChain invocation
        if rm_type is None:
            return search_and_retrieve.invoke({"query": query})
        else:
            return search_and_retrieve.invoke({"query": query, "rm_type": rm_type})

    def _search_concurrent(self, queries: List[str]) -> None:
        """Execute multiple secondary searches concurrently and store results in shared memory."""
        if not queries:
            return

        if len(queries) == 1:
            # Single query - use existing method to avoid overhead
            self._search(queries[0])
            return

        logger.info(f"Executing {len(queries)} searches concurrently")

        # Import here to avoid circular imports
        from src.retrieval.factory import create_retrieval_manager

        try:
            # Get rm_type from config context (same logic as search_and_retrieve tool)
            retrieval_config = (
                self.config_context.retrieval_config
                if hasattr(self, "config_context")
                else None
            )
            rm_type = (
                retrieval_config.retrieval_manager
                if retrieval_config
                else "supabase_faiss"
            )

            # Get cached retrieval manager instance with proper type
            retrieval_manager = create_retrieval_manager(rm_type)

            # Execute concurrent search
            all_results = retrieval_manager.search_concurrent(
                query_list=queries,
                max_results=None,  # Use config defaults
            )

            # Store results in shared memory - we need to simulate the search_and_retrieve tool format
            # Since concurrent search returns flattened results, we'll distribute them among queries
            # For now, we'll create aggregated results for each query
            results_per_query = len(all_results) // len(queries) if all_results else 0

            for i, query in enumerate(queries):
                # Create a search result in the format expected by shared memory
                start_idx = i * results_per_query
                end_idx = (
                    start_idx + results_per_query
                    if i < len(queries) - 1
                    else len(all_results)
                )
                query_results = all_results[start_idx:end_idx]

                # Convert raw retrieval results to ResearchChunk objects (same as search tool)
                import hashlib

                from src.utils.data.models import ResearchChunk

                chunks = []
                for j, result in enumerate(query_results):
                    # Generate chunk ID (same pattern as search tool)
                    chunk_id = f"{self.retrieval_config.retrieval_manager}_{j}_{hashlib.md5(str(result).encode()).hexdigest()[:8]}"

                    if isinstance(result, dict):
                        chunk = ResearchChunk.from_retrieval_result(chunk_id, result)
                        # Add search-specific metadata
                        chunk.metadata.update(
                            {
                                "search_query": query,
                                "relevance_rank": j + 1,
                            }
                        )
                        if "relevance_score" in result:
                            chunk.metadata["relevance_score"] = result.get(
                                "relevance_score"
                            )
                        chunks.append(chunk)

                # Store chunks and get summaries (same as search tool)
                chunk_summaries = self.shared_memory.store_research_chunks(chunks)

                search_result = {
                    "success": True,
                    "query": query,
                    "total_chunks": len(chunk_summaries),
                    "chunk_summaries": chunk_summaries,  # Use correct key name
                }

                # Store in shared memory using the same format as search_and_retrieve tool
                self.shared_memory.store_search_summary(
                    query, search_result
                )  # TODO: NOT CONSISTENT WITH TOOL, REPEATED LOGIC, COULD BE IMPROVED
                logger.debug(
                    f"Stored {len(chunk_summaries)} results for query: {query}"
                )

        except Exception as exc:
            logger.error(f"Concurrent search failed, falling back to sequential: {exc}")
            # Fallback to sequential search
            for query in queries:
                try:
                    self._search(query)
                except Exception as seq_exc:
                    logger.error(
                        f"Sequential fallback failed for query '{query}': {seq_exc}"
                    )

    def _get_best_chunk(self) -> Optional[ResearchChunk]:
        """Get the highest-scoring chunk for context."""
        chunks = self.shared_memory.get_stored_chunks()
        if not chunks:
            logger.warning("No chunks available for context")
            return None

        # Sort by relevance score if available, otherwise take first
        sorted_chunks = sorted(
            chunks, key=lambda x: getattr(x, "relevance_score", 0.0), reverse=True
        )

        top_chunk = sorted_chunks[0]
        logger.info(
            f"Selected top chunk: {top_chunk.chunk_id} (score: {getattr(top_chunk, 'relevance_score', 'N/A')})"
        )
        return top_chunk

    def _generate_queries(
        self, topic: str, top_chunk: Optional[ResearchChunk]
    ) -> List[str]:
        """Generate secondary search queries using LLM reasoning."""
        if not top_chunk:
            logger.warning("No top chunk available, using basic query generation")

        prompt = search_query_generation_prompt(topic, top_chunk, self.num_queries)

        try:
            # Use query generation model
            query_client = self.get_task_client("query_generation")
            response = query_client.call_api(prompt=prompt)
            queries = self._parse_queries_from_response(response)
            logger.info(f"Generated {len(queries)} secondary queries: {queries}")
            return queries
        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            raise RuntimeError(f"Failed to generate secondary queries: {e}") from e

    def _outline(
        self, topic: str, top_chunk: Optional[ResearchChunk], chunk_summaries: str
    ) -> Outline:
        """Create article outline using LLM reasoning with top chunk context."""
        prompt = outline_prompt(topic, top_chunk, chunk_summaries)

        # Use ToM to predict reviewer expectations (only if enabled)
        tom_context = None
        if self.tom_module and self.tom_module.enabled:
            logger.info("Generating ToM prediction for reviewer expectations")
            from src.collaborative.theory_of_mind import AgentRole

            tom_prediction = self.tom_module.predict_agent_response(
                predictor=AgentRole.WRITER,
                target=AgentRole.REVIEWER,
                context={
                    "topic": topic,
                    "has_research": bool(top_chunk),
                    "action": "outline_review",
                },
            )
            tom_context = tom_prediction.reasoning
            logger.info(f"ToM prediction: {tom_context}")

        try:
            # Use outline model for creating outline
            outline_client = self.get_task_client("outline")
            response = outline_client.call_api(
                prompt=self._enhance_prompt_with_tom(prompt, tom_context)
            )
            outline = self._parse_outline_from_response(response)

            # Store outline in memory
            self.shared_memory.state["initial_outline"] = outline
            logger.info(
                f"Created outline with {len(outline.headings)} sections: {outline.headings}"
            )
            return outline
        except Exception as e:
            logger.error(f"LLM outline creation failed: {e}")
            raise RuntimeError(f"Failed to create outline: {e}") from e

    def _write_all_sections(self, outline: Outline):
        """Sequential section writer - clean and reliable."""
        # Get Theory of Mind prediction if enabled
        tom_context = None  # TODO: currently unused
        if self.tom_module and self.tom_module.enabled:
            from src.collaborative.theory_of_mind import AgentRole

            tom_prediction = self.tom_module.predict_agent_response(
                AgentRole.WRITER,
                AgentRole.REVIEWER,
                {
                    "article_metrics": {"sections": len(outline.headings)},
                    "action": "content_review",
                },
            )
            getattr(tom_prediction, "reasoning", None)

        topic = self.shared_memory.state.get("topic")
        chunk_summaries = self.shared_memory.get_search_summaries()
        chunk_summaries_str = (
            "["
            + "; ".join(
                f"{q}: {{"
                + ", ".join(
                    f"{c.get('chunk_id', '')}: {c.get('description', '').strip()}"
                    for c in v.get("chunk_summaries", [])
                )
                + "}}"
                for q, v in chunk_summaries.items()
            )
            + "]"
        )

        # Write sections sequentially - simple and reliable
        article_sections = {}
        logger.info(f"Writing {len(outline.headings)} sections sequentially")

        for i, section in enumerate(outline.headings, 1):
            logger.info(f"Writing section {i}/{len(outline.headings)}: '{section}'")

            try:
                content = self._write_single_section(
                    section, topic, chunk_summaries_str
                )
                if content:
                    article_sections[section] = content
                    logger.info(
                        f"✅ Completed section '{section}' ({len(content)} chars)"
                    )
                else:
                    logger.warning(f"⚠️ Section '{section}' returned no content")

            except Exception as e:
                logger.error(f"❌ Section '{section}' failed: {e}", exc_info=True)
                # Continue with other sections

        # Persist article
        article = Article(
            title=topic,
            content=self._build_full_article_content(topic, article_sections),
            sections=article_sections,
            metadata={"iteration": 0, "sections_count": len(article_sections)},
        )
        self.shared_memory.update_article_state(article)
        # Set article_content for reviewer compatibility
        self.shared_memory.state["article_content"] = article.content
        logger.info(f"Completed writing {len(article_sections)} sections")

    def _write_single_section(
        self, section: str, topic: str, chunk_summaries_str: str
    ) -> Optional[str]:
        """Write a single section with proper error handling."""
        try:
            # 1. Ask LLM to select chunks (Step 6a) - using section selection model
            select_prompt = select_section_chunks_prompt(
                section, topic, chunk_summaries_str, self.num_chunks
            )
            selection_client = self.get_task_client("section_selection")
            select_resp = selection_client.call_api(prompt=select_prompt)

            # 2. Parse and dedupe chunk IDs
            chunk_ids = self.parse_chunk_ids_from_response(select_resp)
            chunk_ids = [
                cid.strip() for cid in chunk_ids if isinstance(cid, str) and cid.strip()
            ]
            seen = set()
            chunk_ids = [c for c in chunk_ids if not (c in seen or seen.add(c))][
                : self.num_chunks
            ]
            if not chunk_ids:
                logger.warning(f"No chunks selected for section '{section}'")
                return None

            # 3. Retrieve chunks via tool
            tool_resp = get_chunks_by_ids.invoke({"chunk_ids": chunk_ids})
            if not tool_resp.get("success"):
                logger.error(f"Tool failed for '{section}': {tool_resp.get('error')}")
                return None

            chunks_obj = tool_resp["chunks"]
            chunk_models = (
                {
                    cid: ResearchChunk.model_validate(data)
                    for cid, data in chunks_obj.items()
                }
                if isinstance(chunks_obj, dict)
                else {
                    d["chunk_id"]: ResearchChunk.model_validate(d) for d in chunks_obj
                }
            )

            # 4. Build chunk string with proper citation info
            chunks_str = ", ".join(
                f"{cid} {{content: {m.content}, "
                f"score: {m.metadata.get('relevance_score')}, url: {m.url or 'N/A'}}}"
                for cid, m in chunk_models.items()
            )

            # 5. Write section content (Step 6b) - using section writing model
            write_prompt = write_section_content_prompt(section, topic, chunks_str)
            writing_client = self.get_task_client("section_writing")
            write_resp = writing_client.call_api(prompt=write_prompt)
            content = self._clean_section_content(write_resp)

            return content

        except Exception as e:
            logger.error(f"Failed to write section '{section}': {e}", exc_info=True)
            return None

    @staticmethod
    def parse_chunk_ids_from_response(response: str) -> List[str]:
        """Parse chunk IDs from LLM response."""
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        chunk_ids = []

        for line in lines:
            # Assume each line is a chunk ID or comma-separated IDs
            ids = [part.strip() for part in line.split(",") if part.strip()]
            chunk_ids.extend(ids)

        return chunk_ids if chunk_ids else []

    def _get_pending_feedback(self) -> Dict:
        """Get pending feedback using tool."""
        self._log_tool_decision(
            "get_feedback", "Retrieve pending feedback", {"only_pending": True}
        )
        result = self.feedback_tool.invoke({"only_pending": True})

        if not result.get("success"):
            raise RuntimeError(f"Failed to retrieve pending feedback: {result}")
        return result

    def _extract_target_sections(self, feedback_result: Dict) -> List[str]:
        """Extract sections that need revision from feedback (Python logic)."""
        feedback_items = feedback_result.get("feedback", [])
        sections_to_revise = set()

        for feedback_item in feedback_items:
            target_section = feedback_item.get("target_section")
            if target_section and feedback_item.get("status") == "pending":
                sections_to_revise.add(target_section)

        return list(sections_to_revise)

    def _revise_section(self, section: str):
        """Revise a specific section with feedback and ToM context."""
        # Get section-specific feedback
        section_feedback = self._get_section_feedback(section)

        # Get previous section content
        previous_iteration = self.shared_memory.get_current_iteration() - 1
        previous_sections = self.shared_memory.get_sections_from_iteration(
            previous_iteration
        )
        previous_content = previous_sections.get(section) if previous_sections else None

        if not previous_content:
            logger.warning(
                f"No previous content found for section '{section}', creating new content"
            )

        # Get relevant chunks as ResearchChunk objects
        relevant_chunks = self.shared_memory.get_stored_chunks()

        # Use ToM to predict reviewer response (only if enabled)
        tom_context = None
        if self.tom_module and self.tom_module.enabled:
            logger.info(f"Generating ToM prediction for section '{section}' revision")
            from src.collaborative.theory_of_mind import AgentRole

            tom_prediction = self.tom_module.predict_agent_response(
                predictor=AgentRole.WRITER,
                target=AgentRole.REVIEWER,
                context={
                    "section": section,
                    "has_previous_content": bool(previous_content),
                    "feedback_count": len(section_feedback),
                    "action": "section_revision_review",
                },
            )
            tom_context = tom_prediction.reasoning

        # Generate revised content
        revised_content = self._revise_section_content(
            section, previous_content, section_feedback, relevant_chunks, tom_context
        )

        # Update section in current article
        current_article = self.shared_memory.get_current_article()
        logger.info(
            f"🔍 REVISION DEBUG: current_article is {'None' if current_article is None else 'available'} in iteration {self.shared_memory.get_current_iteration()}"
        )

        if not current_article:
            # During revision, get article from previous iteration if current iteration has no draft
            current_iteration = self.shared_memory.get_current_iteration()
            if current_iteration > 0:
                # Get draft from previous iteration
                prev_draft = self.shared_memory.get_previous_draft()
                if prev_draft:
                    logger.info(
                        f"🔍 REVISION DEBUG: Using draft from iteration {current_iteration - 1} for revision"
                    )
                    prev_sections = self.shared_memory.get_sections_from_iteration(
                        current_iteration - 1
                    )
                    from src.utils.data import Article

                    current_article = Article(
                        title=self.shared_memory.state.get("topic", "Untitled"),
                        content=prev_draft,
                        sections=prev_sections,
                        metadata=self.shared_memory.state.get("metadata", {}),
                    )

        if current_article:
            current_article.sections[section] = revised_content
            # Rebuild full content
            current_article.content = self._build_full_article_content(
                current_article.title, current_article.sections
            )
            self.shared_memory.update_article_state(current_article)
            # Set article_content for reviewer compatibility
            self.shared_memory.state["article_content"] = current_article.content
            logger.info(
                f"🔍 REVISION DEBUG: Successfully stored revised article in iteration {self.shared_memory.get_current_iteration()}"
            )
        else:
            logger.error(
                f"🔍 REVISION DEBUG: No article available for revision in iteration {self.shared_memory.get_current_iteration()}"
            )

        logger.info(f"Revised section '{section}' successfully")

    def _mark_feedback_addressed(self, sections_to_revise: List[str]):
        """Mark feedback as addressed for revised sections."""
        for section in sections_to_revise:
            section_feedback_items = self.shared_memory.get_feedback_for_section(
                section
            )
            for feedback_item in section_feedback_items:
                if feedback_item.get("status") == "pending":
                    feedback_id = feedback_item.get("feedback_id")
                    if feedback_id:
                        # Use existing SharedMemory method to mark as claimed by writer
                        self.shared_memory.mark_feedback_claimed_by_writer(
                            feedback_id, f"Addressed in section revision"
                        )
                        logger.info(
                            f"Marked feedback {feedback_id} as addressed for section '{section}'"
                        )

    def _get_section_feedback(self, section: str) -> List[Dict]:
        """Get feedback specific to a section."""
        return self.shared_memory.get_feedback_for_section(section)

    def _build_revision_prompt(
        self,
        section: str,
        previous_content: Optional[str],
        feedback: List[Dict],
        chunks: List[ResearchChunk],
        tom_context: Optional[str] = None,
    ) -> str:
        """Build revision prompt for section content based on feedback."""
        prompt_parts = [
            f"You are an expert writer revising the '{section}' section of an article.",
            "",
            "TASK: Revise the section content based on the feedback provided.",
            "",
        ]

        if previous_content:
            prompt_parts.extend(
                [
                    "PREVIOUS CONTENT:",
                    previous_content,
                    "",
                ]
            )

        if feedback:
            prompt_parts.append("FEEDBACK TO ADDRESS:")
            for i, fb in enumerate(feedback, 1):
                prompt_parts.append(
                    f"{i}. {fb.get('feedback', 'No feedback provided')}"
                )
            prompt_parts.append("")

        if chunks:
            prompt_parts.extend(
                [
                    "AVAILABLE RESEARCH SOURCES:",
                    'Use these sources for citations with format <c cite="chunk_id"/>:',
                ]
            )
            for chunk in chunks[:5]:  # Limit to top 5 chunks
                prompt_parts.append(f"- {chunk.chunk_id}: {chunk.content[:200]}...")
            prompt_parts.append("")

        if tom_context:
            prompt_parts.extend(
                [
                    "REVIEWER EXPECTATIONS:",
                    tom_context,
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "REQUIREMENTS:",
                "- Address all feedback points",
                "- Maintain factual accuracy",
                '- Use proper citations with <c cite="chunk_id"/> format',
                "- Keep content well-structured and engaging",
                "",
                f"Write the revised content for the '{section}' section:",
            ]
        )

        return "\n".join(prompt_parts)

    def _revise_section_content(
        self,
        section: str,
        previous_content: Optional[str],
        feedback: List[Dict],
        chunks: List[ResearchChunk],
        tom_context: Optional[str] = None,
    ) -> str:
        """Revise section content using LLM with feedback and ToM context."""
        prompt = self._build_revision_prompt(
            section, previous_content, feedback, chunks, tom_context
        )

        try:
            # Use section revision model for revising sections (Iteration 1+)
            revision_client = self.get_task_client("section_revision")
            response = revision_client.call_api(prompt=prompt)
            return self._clean_section_content(response)
        except Exception as e:
            logger.error(
                f"LLM revision failed for section '{section}': {e}", exc_info=True
            )
            raise RuntimeError(f"Failed to revise section '{section}': {e}") from e

    def _enhance_prompt_with_tom(
        self, base_prompt: str, tom_context: Optional[str]
    ) -> str:
        """Enhance prompt with Theory of Mind context if available."""
        if not tom_context:
            return base_prompt
        return f"{base_prompt}\n\nCollaborative context: {tom_context}"

    @staticmethod
    def _parse_queries_from_response(response: str) -> List[str]:
        """Parse queries from LLM response."""
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        # Filter out empty lines and take up to 5 queries
        queries = [line for line in lines if line and not line.startswith("#")][:5]
        return queries if queries else ["No additional queries generated"]

    def _parse_outline_from_response(self, response: str) -> Outline:
        """Parse outline from LLM response using V1-style ## headings."""
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        headings = []

        for line in lines:
            clean_line = line.strip()

            # Look for V1-style ## section headings
            if clean_line.startswith("## "):
                heading = clean_line[3:].strip()  # Remove "## " prefix
                # Clean any md formatting
                heading = heading.replace("**", "").replace("*", "").strip()
                # Remove brackets if present [Short Title]
                if heading.startswith("[") and heading.endswith("]"):
                    heading = heading[1:-1].strip()

                if heading and len(heading) > 3:  # Ensure meaningful headings
                    headings.append(heading)

        if not headings:
            # Fallback outline with proper sections
            logger.error(f"Failed to parse outline from response: {response}")
            raise RuntimeError(f"Failed to parse outline from response: {response}")

        topic = self.shared_memory.state.get("topic", "Topic")
        logger.info(f"Parsed {len(headings)} main sections from outline: {headings}")
        return Outline(title=topic, headings=headings)

    def _clean_section_content(self, content: str) -> str:
        """Clean and format section content."""
        # Remove common artifacts and clean up
        content = content.strip()

        # Remove section headers if they appear at the start
        lines = content.split("\n")
        cleaned_lines = []

        for line in lines:
            # Skip lines that look like section headers at the start
            if not cleaned_lines and (
                line.startswith("#") or line.isupper() and len(line) < 50
            ):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _build_full_article_content(self, title: str, sections: dict) -> str:
        """Build complete article content with citations and references."""
        content_parts = []

        # Add title
        content_parts.append(f"# {title}\n")

        # Build main content from sections dictionary
        main_content = ""
        for section_title, section_content in sections.items():
            main_content += f"\n## {section_title}\n\n{section_content}\n"

        content_parts.append(main_content)

        # Extract citations and build reference list
        citations = self._extract_citations_from_content(main_content)
        references = self._build_reference_list(citations)

        if references:
            content_parts.append(references)

        return "".join(content_parts)

    def _extract_citations_from_content(
        self, content: str
    ) -> Dict[str, Dict[str, str]]:
        """Extract all chunk ID citations from content and get their URLs."""
        import re

        citations = {}

        # Find all chunk ID citations in citation tags
        citation_pattern = r'<c cite="([a-zA-Z0-9_]+)"/>'
        matches = re.findall(citation_pattern, content)

        if not matches:
            return citations

        # Get chunk details for each citation
        try:
            from src.collaborative.tools.tool_definitions import get_chunks_by_ids

            result = get_chunks_by_ids.invoke({"chunk_ids": list(set(matches))})

            if result.get("success") and result.get("chunks"):
                chunks_data = result["chunks"]

                for chunk_id in set(matches):
                    if chunk_id in chunks_data:
                        chunk_info = chunks_data[chunk_id]
                        citations[chunk_id] = {
                            "url": chunk_info.get("url", "N/A"),
                            "source": chunk_info.get("source", "Unknown"),
                            "title": chunk_info.get("title", "Unknown"),
                        }
        except Exception as e:
            logger.warning(f"Failed to extract citation details: {e}")

        return citations

    def _build_reference_list(self, citations: Dict[str, Dict[str, str]]) -> str:
        """Build a formatted reference list from citations."""
        if not citations:
            return ""

        ref_content = "\n\n## References\n\n"

        for i, (chunk_id, details) in enumerate(citations.items(), 1):
            url = details.get("url", "N/A")
            title = details.get("title", "Unknown")
            source = details.get("source", "Unknown")

            if url != "N/A":
                ref_content += f"{i}. {title} - {url}\n"
            else:
                ref_content += f"{i}. {title} (Source: {source})\n"

        return ref_content

    def _log_tool_decision(
        self, tool_name: str, decision_reason: str, context: dict = None
    ):
        """Log tool selection with decision reasoning for debugging."""
        log_msg = f"Tool decision: {tool_name} - Reason: {decision_reason}"
        if context:
            log_msg += f" - Context: {context}"
        logger.info(log_msg)
