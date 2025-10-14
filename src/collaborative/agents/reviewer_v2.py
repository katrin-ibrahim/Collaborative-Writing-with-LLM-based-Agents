# src/collaborative/agents/reviewer_v2.py
"""
ReviewerV2 Agent Implementation - New Architecture

Implements a deterministic citation processor that handles:
1. Citation tag extraction from article text
2. Reference number mapping and enumeration
3. Reference section assembly using chunk metadata
4. Validation of citations and sources

Design principles:
- Deterministic parsing (no LLM for citation processing)
- Optional LLM verification mode for semantic checking
- Stable reference numbering system
- Clean separation of concerns
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from src.collaborative.agents.base_agent import BaseAgent
from src.collaborative.agents.templates import build_holistic_review_prompt
from src.collaborative.tools.tool_definitions import (
    get_article_metrics,
    get_chunks_by_ids,
)
from src.config.config_context import ConfigContext
from src.utils.data import Article

logger = logging.getLogger(__name__)


class ReviewerV2(BaseAgent):
    """
    Deterministic citation processor and validator.

    New Architecture Workflow:
    1. Extract citation tags from article text
    2. Create stable reference number mapping
    3. Replace citation tags with numbered references
    4. Assemble reference section using chunk metadata
    5. Validate citations and flag issues
    6. Optional: LLM-based semantic verification
    """

    def __init__(self):
        super().__init__()
        self.shared_memory = ConfigContext.get_memory_instance()
        if not self.shared_memory:
            raise RuntimeError("SharedMemory instance not found in ConfigContext")

        # Tool for chunk metadata retrieval
        self.chunks_tool = get_chunks_by_ids

    def process(self) -> None:
        """Main entry point for deterministic citation processing."""
        iteration = self.shared_memory.get_current_iteration()
        logger.info(
            f"ReviewerV2 starting citation processing for iteration {iteration}"
        )

        self._execute_citation_workflow()

    def _execute_citation_workflow(self):
        """
        New deterministic workflow:
        1. Extract citation tags → 2. Create reference mapping →
        3. Replace tags with numbers → 4. Assemble references → 5. Validate → 6. Update article
        """

        # Step 1: Prepare article from memory
        logger.info("Step 1: Preparing article from memory")
        try:
            article = self._prepare_article_from_memory()
            logger.info(
                f"Prepared article: '{article.title}' ({len(article.content)} chars)"
            )
        except Exception as e:
            logger.error(f"Failed to prepare article: {e}")
            raise RuntimeError(f"ReviewerV2 step 'prepare_article' failed: {e}") from e

        # Step 2: Extract citation tags (deterministic parsing)
        logger.info("Step 2: Extracting citation tags")
        try:
            structured_claims = self._extract_citation_tags(article)
            logger.info(f"Extracted {len(structured_claims)} citation claims")
        except Exception as e:
            logger.error(f"Failed to extract citation tags: {e}")
            raise RuntimeError(f"ReviewerV2 step 'extract_tags' failed: {e}") from e

        # Step 3: Create stable reference mapping (deterministic)
        logger.info("Step 3: Creating reference mapping")
        try:
            ref_map, citation_order = self._create_reference_mapping(structured_claims)
            logger.info(f"Created reference map with {len(ref_map)} unique citations")
        except Exception as e:
            logger.error(f"Failed to create reference mapping: {e}")
            raise RuntimeError(
                f"ReviewerV2 step 'reference_mapping' failed: {e}"
            ) from e

        # Step 4: Replace citation tags with numbered references (deterministic)
        logger.info("Step 4: Replacing citation tags with numbered references")
        try:
            updated_content = self._replace_citation_tags_with_numbers(
                article.content, ref_map
            )
            logger.info("Successfully replaced citation tags with reference numbers")
        except Exception as e:
            logger.error(f"Failed to replace citation tags: {e}")
            raise RuntimeError(f"ReviewerV2 step 'replace_tags' failed: {e}") from e

        # Step 5: Assemble reference section (deterministic + tool call)
        logger.info("Step 5: Assembling reference section")
        try:
            reference_section = self._assemble_reference_section(
                citation_order, ref_map
            )
            logger.info(
                f"Assembled reference section with {len(citation_order)} references"
            )
        except Exception as e:
            logger.error(f"Failed to assemble reference section: {e}")
            raise RuntimeError(
                f"ReviewerV2 step 'assemble_references' failed: {e}"
            ) from e

        # Step 6: Validate citations (deterministic)
        logger.info("Step 6: Validating citations")
        try:
            validation_results = self._validate_citations(structured_claims, ref_map)
            logger.info(f"Validation completed: {validation_results}")
        except Exception as e:
            logger.error(f"Failed to validate citations: {e}")
            raise RuntimeError(
                f"ReviewerV2 step 'validate_citations' failed: {e}"
            ) from e

        # Step 7: Update article with processed content and references
        logger.info("Step 7: Updating article with processed content")
        try:
            final_content = updated_content + reference_section
            self._update_article_in_memory(article, final_content, validation_results)
            logger.info("Successfully updated article with citations and references")
        except Exception as e:
            logger.error(f"Failed to update article: {e}")
            raise RuntimeError(f"ReviewerV2 step 'update_article' failed: {e}") from e

        # Step 8: Generate LLM-based review feedback (semantic analysis)
        logger.info("Step 8: Generating LLM-based review feedback")
        try:
            self._generate_llm_review_feedback(article, validation_results)
            logger.info("Successfully generated review feedback")
        except Exception as e:
            logger.error(f"Failed to generate review feedback: {e}")
            raise RuntimeError(
                f"ReviewerV2 step 'generate_feedback' failed: {e}"
            ) from e

        logger.info("ReviewerV2 hybrid processing completed successfully")

    def _prepare_article_from_memory(self) -> Article:
        """Prepare article object from shared memory."""
        # Get current draft content and sections
        current_draft = self.shared_memory.get_current_draft()
        if not current_draft:
            raise RuntimeError("No current draft found in SharedMemory")

        # Get current sections
        current_iteration = self.shared_memory.get_current_iteration()
        sections = self.shared_memory.get_sections_from_iteration(current_iteration)
        if not sections:
            sections = {}

        # Get topic as title
        topic = self.shared_memory.state.get("topic", "Unknown Topic")

        # Build article object
        article = Article(
            title=topic,
            content=current_draft,
            sections=sections,
            metadata=self.shared_memory.state.get("metadata", {}),
        )

        logger.info(
            f"Prepared article: '{article.title}' with {len(article.sections)} sections"
        )
        return article

    def _extract_citation_tags(self, article: Article) -> List[Dict]:
        """
        Extract all <c cite="..."/> tags from article text and create structured claims.
        Returns: List[Dict] with section, sentence, chunks mapping
        """
        structured_claims = []

        # Find all citation tags in the content
        citation_pattern = r'<c cite="([^"]+)"/>'

        # Split content by sections to track which section each citation belongs to
        sections = (
            article.sections if article.sections else {"General": article.content}
        )

        for section_name, section_content in sections.items():
            # Split into sentences for better granularity
            sentences = self._split_into_sentences(section_content)

            for sentence in sentences:
                # Find citations in this sentence
                matches = re.findall(citation_pattern, sentence)
                if matches:
                    # Clean sentence of citation tags for storage
                    clean_sentence = re.sub(citation_pattern, "", sentence).strip()
                    structured_claims.append(
                        {
                            "section": section_name,
                            "sentence": clean_sentence,
                            "chunks": matches,
                            "original_sentence": sentence,  # Keep original for tag replacement
                        }
                    )

        return structured_claims

    def _create_reference_mapping(
        self, structured_claims: List[Dict]
    ) -> Tuple[Dict[str, int], List[str]]:
        """
        Create stable enumeration mapping: chunk_id -> reference_number
        Returns: (ref_map, citation_order)
        """
        ref_map = {}
        citation_order = []
        next_n = 1

        # Process claims in order to maintain stable enumeration
        for claim in structured_claims:
            for chunk_id in claim["chunks"]:
                if chunk_id not in ref_map:
                    ref_map[chunk_id] = next_n
                    citation_order.append(chunk_id)
                    next_n += 1

        return ref_map, citation_order

    def _replace_citation_tags_with_numbers(
        self, content: str, ref_map: Dict[str, int]
    ) -> str:
        """Replace <c cite="chunk_id"/> with [reference_number] in content."""

        def replace_citation(match):
            chunk_id = match.group(1)
            ref_num = ref_map.get(chunk_id, "?")
            return f"[{ref_num}]"

        citation_pattern = r'<c cite="([^"]+)"/>'
        updated_content = re.sub(citation_pattern, replace_citation, content)

        return updated_content

    def _assemble_reference_section(
        self, citation_order: List[str], ref_map: Dict[str, int]
    ) -> str:
        """
        Generate formatted reference section using chunk metadata.
        Returns: Formatted reference section string
        """
        if not citation_order:
            return ""

        # Get chunk metadata for all citations
        try:
            result = self.chunks_tool.invoke({"chunk_ids": citation_order})
            if not result.get("success"):
                logger.error(f"Failed to retrieve chunk metadata: {result}")
                return "\n\n## References\n\n[Reference details unavailable]\n"

            chunks_data = result.get("chunks", {})
        except Exception as e:
            logger.error(f"Error retrieving chunk metadata: {e}")
            return "\n\n## References\n\n[Reference details unavailable]\n"

        # Build reference section
        ref_content = "\n\n## References\n\n"

        for chunk_id in citation_order:
            ref_num = ref_map[chunk_id]

            if chunk_id in chunks_data:
                chunk_info = chunks_data[chunk_id]
                url = chunk_info.get("url", "N/A")
                title = chunk_info.get("title", "Unknown")
                source = chunk_info.get("source", "Unknown")

                if url != "N/A":
                    ref_content += f"{ref_num}. {title} - {url}\n"
                else:
                    ref_content += f"{ref_num}. {title} (Source: {source})\n"
            else:
                ref_content += f"{ref_num}. [Chunk {chunk_id} - metadata unavailable]\n"

        return ref_content

    def _validate_citations(
        self, structured_claims: List[Dict], ref_map: Dict[str, int]
    ) -> Dict:
        """
        Validate citations and flag issues.
        Returns: Validation results with stats and warnings
        """
        validation_results = {
            "total_citations": len(ref_map),
            "valid_citations": 0,
            "dangling_citations": [],
            "missing_chunks": [],
            "needs_source_count": 0,
            "duplicate_citations": 0,
        }

        # Check for needs_source tags
        needs_source_pattern = r"<needs_source/>"
        article_content = self.shared_memory.get_current_draft() or ""
        needs_source_matches = re.findall(needs_source_pattern, article_content)
        validation_results["needs_source_count"] = len(needs_source_matches)

        # Validate each citation exists in chunks
        try:
            result = self.chunks_tool.invoke({"chunk_ids": list(ref_map.keys())})
            if result.get("success"):
                chunks_data = result.get("chunks", {})
                for chunk_id in ref_map.keys():
                    if chunk_id in chunks_data:
                        validation_results["valid_citations"] += 1
                    else:
                        validation_results["missing_chunks"].append(chunk_id)
            else:
                logger.warning("Could not validate chunk existence")
        except Exception as e:
            logger.warning(f"Citation validation failed: {e}")

        # Check for duplicate citations (same chunk cited multiple times)
        chunk_counts = {}
        for claim in structured_claims:
            for chunk_id in claim["chunks"]:
                chunk_counts[chunk_id] = chunk_counts.get(chunk_id, 0) + 1

        validation_results["duplicate_citations"] = sum(
            1 for count in chunk_counts.values() if count > 1
        )

        logger.info(
            f"Citation validation: {validation_results['valid_citations']}/{validation_results['total_citations']} valid, "
            f"{len(validation_results['missing_chunks'])} missing, "
            f"{validation_results['needs_source_count']} need sources"
        )

        return validation_results

    def _update_article_in_memory(
        self, article: Article, final_content: str, validation_results: Dict
    ):
        """Update article in shared memory with processed content and validation metadata."""
        # Update article content
        article.content = final_content
        article.metadata = article.metadata or {}
        article.metadata.update(
            {
                "citation_validation": validation_results,
                "processed_by_reviewer": True,
                "processing_timestamp": self.shared_memory.get_current_iteration(),
            }
        )

        # Store updated article
        self.shared_memory.update_article_state(article)
        self.shared_memory.state["article_content"] = final_content

        logger.info("Updated article with processed citations and references")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting for citation extraction."""
        # Simple sentence splitting - can be improved with proper NLP if needed
        sentences = []
        current_sentence = ""

        for char in text:
            current_sentence += char
            if char in ".!?" and len(current_sentence.strip()) > 10:
                sentences.append(current_sentence.strip())
                current_sentence = ""

        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return sentences

    # ======================== LLM-BASED REVIEW METHODS ========================
    # These methods handle the semantic review and feedback generation

    def _generate_llm_review_feedback(self, article: Article, validation_results: Dict):
        """
        Generate LLM-based review feedback after citation processing.
        This is the semantic review component that stays LLM-based.
        """
        logger.info("Starting LLM-based review feedback generation")

        # Step 1: Get article metrics
        try:
            metrics_result = get_article_metrics.invoke(
                {"content": article.content, "title": article.title}
            )
            if not metrics_result.get("success"):
                raise RuntimeError(f"Failed to get article metrics: {metrics_result}")
            metrics = metrics_result.get("metrics", {})
            logger.info(f"Article metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to get article metrics: {e}")
            metrics = {"word_count": 0, "section_count": 0}

        # Step 2: Determine review strategy
        review_strategy = self._determine_review_strategy(metrics, validation_results)
        logger.info(f"Selected review strategy: {review_strategy}")

        # Step 3: Generate ToM context if available
        tom_context = None
        if (
            hasattr(self, "shared_memory")
            and self.shared_memory.tom_module
            and self.shared_memory.tom_module.enabled
        ):
            logger.info("Generating ToM prediction for writer response")
            try:
                from src.collaborative.theory_of_mind import AgentRole

                tom_prediction = self.shared_memory.tom_module.predict_agent_response(
                    predictor=AgentRole.REVIEWER,
                    target=AgentRole.WRITER,
                    context={
                        "review_strategy": review_strategy,
                        "article_metrics": metrics,
                        "citation_validation": validation_results,
                        "iteration": self.shared_memory.get_current_iteration(),
                        "action": "feedback_response",
                    },
                )
                tom_context = tom_prediction.reasoning
                logger.info(f"ToM prediction: {tom_context}")
            except Exception as e:
                logger.warning(f"ToM prediction failed: {e}")

        # Step 4: Generate holistic review with section-specific feedback
        try:
            fake_fact_check = {
                "verified_claims": [],
                "unverified_claims": [],
            }  # We don't do fact-checking in this new architecture
            section_feedback_map = self._generate_holistic_review_with_section_feedback(
                article, metrics, fake_fact_check, tom_context, review_strategy
            )
            logger.info(f"Generated feedback for {len(section_feedback_map)} sections")
        except Exception as e:
            logger.error(f"Failed to generate section feedback: {e}")
            section_feedback_map = {}

        # Step 5: Store section-specific feedback
        try:
            self._store_section_feedback(section_feedback_map)
            logger.info("Successfully stored section feedback")
        except Exception as e:
            logger.error(f"Failed to store section feedback: {e}")

    def _determine_review_strategy(
        self, metrics: Dict, validation_results: Dict
    ) -> str:
        """
        Determine review approach based on article metrics and citation validation.
        """
        word_count = metrics.get("word_count", 0)
        section_count = metrics.get("section_count", 0)
        iteration = self.shared_memory.get_current_iteration()

        # Consider citation issues in strategy
        missing_chunks = len(validation_results.get("missing_chunks", []))
        needs_source_count = validation_results.get("needs_source_count", 0)

        # Citation-focused if many citation issues
        if missing_chunks > 2 or needs_source_count > 3:
            logger.info(
                f"Selected citation-focused strategy (missing: {missing_chunks}, needs_source: {needs_source_count})"
            )
            return "citation-focused"

        # Expansion-focused for short articles
        if word_count < 1000 or section_count < 3:
            logger.info(
                f"Selected expansion-focused strategy (word_count: {word_count}, sections: {section_count})"
            )
            return "expansion-focused"

        # Accuracy-focused for first iteration
        if iteration == 0:
            logger.info(f"Selected accuracy-focused strategy (iteration: {iteration})")
            return "accuracy-focused"

        # Holistic review for mature articles
        logger.info("Selected holistic strategy for mature article")
        return "holistic"

    def _generate_holistic_review_with_section_feedback(
        self,
        article: Article,
        metrics: Dict,
        fact_check_results: Dict,
        tom_context: Optional[str],
        review_strategy: str,
    ) -> Dict[str, str]:
        """
        Generate holistic review with section-specific feedback using LLM.
        """
        prompt = build_holistic_review_prompt(
            article, metrics, fact_check_results, tom_context, review_strategy
        )

        try:
            # Use review model for holistic review and section feedback generation (Step 8)
            review_client = self.get_task_client("review")
            response = review_client.call_api(prompt=prompt)
            section_feedback_map = self._parse_section_feedback_response(
                response, article
            )
            logger.info(f"Generated feedback for {len(section_feedback_map)} sections")
            return section_feedback_map
        except Exception as e:
            logger.error(f"Holistic review generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate holistic review: {e}") from e

    def _store_section_feedback(self, section_feedback_map: Dict[str, str]):
        """Store section-specific feedback using existing memory methods."""
        iteration = self.shared_memory.get_current_iteration()

        for section_name, feedback_text in section_feedback_map.items():
            if feedback_text.strip():  # Only store non-empty feedback
                self.shared_memory.store_structured_feedback(
                    feedback_text=feedback_text,
                    target_section=section_name,
                    iteration=iteration,
                )
                logger.info(f"Stored feedback for section: {section_name}")

    def _parse_section_feedback_response(
        self, response: str, article: Article
    ) -> Dict[str, str]:
        """Parse section feedback from LLM response."""
        section_feedback = {}

        # Look for section feedback pattern
        lines = response.split("\n")
        current_section = None
        current_feedback = []

        in_section_feedback = False

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for section feedback start
            if "SECTION FEEDBACK" in line.upper():
                in_section_feedback = True
                logger.info(f"Found SECTION FEEDBACK at line {i}")
                continue

            if not in_section_feedback:
                continue

            # Check if this line is a section header using outline-style format: ## Section Name
            is_section_header = False
            clean_section_name = None

            # Look for ## headings
            if line.startswith("## "):
                clean_section_name = line[3:].strip()  # Remove "## " prefix
                # Clean any md formatting
                clean_section_name = (
                    clean_section_name.replace("**", "").replace("*", "").strip()
                )
                # Remove brackets if present [Short Title]
                if clean_section_name.startswith("[") and clean_section_name.endswith(
                    "]"
                ):
                    clean_section_name = clean_section_name[1:-1].strip()

                if (
                    clean_section_name and len(clean_section_name) > 3
                ):  # Ensure meaningful headings
                    is_section_header = True
                    logger.info(
                        f"Found outline-style section header: '{clean_section_name}'"
                    )

            if is_section_header:
                # Save previous section if exists
                if current_section and current_feedback:
                    feedback_text = "\n".join(current_feedback).strip()
                    if feedback_text and not feedback_text.startswith("["):
                        section_feedback[current_section] = feedback_text
                        logger.info(f"Saved section '{current_section}' with feedback")

                # Start new section
                current_section = clean_section_name
                current_feedback = []
                logger.info(f"Started new section: '{current_section}'")
            elif current_section and line:
                # Add to current section feedback
                current_feedback.append(line)

        # Save final section
        if current_section and current_feedback:
            feedback_text = "\n".join(current_feedback).strip()
            if feedback_text and not feedback_text.startswith("["):
                section_feedback[current_section] = feedback_text

        # If no sections found, try to extract any feedback as general feedback
        if not section_feedback and article and article.sections:
            first_section = list(article.sections.keys())[0]
            # Extract meaningful feedback from response
            meaningful_lines = [
                line
                for line in lines
                if line and not line.startswith("[") and len(line) > 20
            ]
            if meaningful_lines:
                section_feedback[first_section] = "\n".join(meaningful_lines[:3])

        return section_feedback
