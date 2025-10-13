# src/collaborative/tools/tools.py
"""
Core tool definitions for collaborative agents.
"""

import hashlib

import logging
import re
from langchain_core.tools import tool
from typing import Any, Dict

from src.config.config_context import ConfigContext
from src.retrieval.factory import create_retrieval_manager
from src.utils.data import ResearchChunk

logger = logging.getLogger(__name__)


@tool
def search_and_retrieve(query: str, rm_type: str = None) -> Dict[str, Any]:
    """
    Search external sources, store chunks in memory, and return summaries for context.

    WORKFLOW: Searches → stores in memory → returns chunk summaries and IDs.
    Use get_chunks_by_ids() to retrieve full content when needed.

    USAGE:
    - Research: search_and_retrieve("topic keywords")
    - Follow-up: Use returned chunk_ids with get_chunks_by_ids() for full content

    Args:
        query: Search query for external sources
        rm_type: Retrieval manager ("wiki", "supabase_faiss") - defaults to config setting

    Returns:
        Dictionary with chunk_summaries (id+summary pairs) and success status
    """
    try:
        if not query or not query.strip():
            return {
                "query": query,
                "chunk_summaries": [],
                "total_chunks": 0,
                "message": "Empty query provided",
                "success": False,
            }

        # Get memory instance
        memory = ConfigContext.get_memory_instance()
        if not memory:
            return {
                "query": query,
                "chunk_summaries": [],
                "total_chunks": 0,
                "message": "Memory not initialized",
                "success": False,
                "error": "Memory system not available",
            }

        # Get retrieval manager from config if not explicitly provided
        if rm_type is None:
            retrieval_config = ConfigContext.get_retrieval_config()
            rm_type = (
                retrieval_config.retrieval_manager
                if retrieval_config
                else "supabase_faiss"
            )

        # Get retrieval manager
        retrieval_manager = create_retrieval_manager(rm_type)

        # Perform search - RM returns List[Dict] or List[str]
        raw_results = retrieval_manager.search(
            query_or_queries=query.strip(),
            topic=query,  # Use query as topic for context
        )

        # Process results: create ResearchChunk objects
        chunks = []
        for i, result in enumerate(raw_results):
            # Generate simple chunk ID
            chunk_id = f"search_{rm_type}_{i}_{hashlib.md5(str(result).encode()).hexdigest()[:8]}"

            # Create ResearchChunk from retrieval result (handles all format variations)
            if isinstance(result, dict):
                chunk = ResearchChunk.from_retrieval_result(chunk_id, result)
                # Add search-specific metadata
                meta_update = {
                    "rm_type": rm_type,
                    "search_query": query,
                    "relevance_rank": i + 1,
                }
                # Persist retriever similarity score if available
                if "relevance_score" in result:
                    meta_update["relevance_score"] = result.get("relevance_score")
                chunk.metadata.update(meta_update)
            else:
                # Fallback for unexpected formats (should not happen with current RMs)
                content = str(result)
                words = content.split()
                description = (
                    " ".join(words[:25]) + "..." if len(words) > 25 else content
                )
                chunk = ResearchChunk(
                    chunk_id=chunk_id,
                    description=description,
                    content=content,
                    source=f"{rm_type}_{i+1}",
                    metadata={
                        "rm_type": rm_type,
                        "search_query": query,
                        "relevance_rank": i + 1,
                    },
                )

            chunks.append(chunk)

        # Store filtered chunks in memory and build summaries
        chunk_summaries = memory.store_research_chunks(chunks)

        # Store search summary in memory for later use
        search_result = {
            "query": query,
            "rm_type": rm_type,
            "chunk_summaries": chunk_summaries,
            "total_chunks": len(chunk_summaries),
            "message": f"Found and stored {len(chunk_summaries)} chunks for '{query}'. Use get_chunks_by_ids to retrieve full content.",
            "success": True,
        }
        memory.store_search_summary(query, search_result)

        logger.info(
            f"Search completed: '{query}' → {len(chunk_summaries)} chunks stored in memory"
        )

        return search_result

    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return {
            "query": query,
            "rm_type": rm_type,
            "chunk_summaries": [],
            "total_chunks": 0,
            "message": f"Search failed: {e}",
            "success": False,
            "error": str(e),
        }


@tool
def get_article_metrics(content: str, title: str) -> Dict[str, Any]:
    """Get objective metrics about article structure."""
    try:
        metrics = {
            "word_count": len(content.split()),
            "heading_count": len(re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)),
            "paragraph_count": len([p for p in content.split("\n\n") if p.strip()]),
        }

        return {"success": True, "metrics": metrics}
    except Exception as e:
        return {"success": False, "error": str(e), "metrics": {}}


# =================== Memory Tools ===================


@tool
def get_chunks_by_ids(chunk_ids) -> Dict[str, Any]:
    """
    Retrieve full research content by chunk IDs for detailed analysis or writing.

    ESSENTIAL WORKFLOW TOOL: Get complete content after using search_and_retrieve.
    The main way to access actual content for reading, analysis, and writing.

    Args:
        chunk_ids: List of chunk IDs from search results, or a comma-separated string
    """
    if isinstance(chunk_ids, str):
        req_ids = [cid.strip() for cid in chunk_ids.split(",") if cid.strip()]
    elif isinstance(chunk_ids, list):
        # Validate list items are strings
        bad = [x for x in chunk_ids if not isinstance(x, str)]
        if bad:
            return {
                "success": False,
                "error": f"All chunk_ids must be strings. Bad values: {bad}",
            }
        # De-dup while preserving order
        seen = set()
        req_ids = [x for x in chunk_ids if not (x in seen or seen.add(x))]
    else:
        return {
            "success": False,
            "error": "chunk_ids must be a list of strings or a comma-separated string",
        }

    memory = ConfigContext.get_memory_instance()
    if not memory:
        return {"success": False, "error": "Memory not initialized"}

    try:
        # Expecting: Dict[str, ResearchChunk]
        chunks = memory.get_chunks_by_ids(req_ids)

        # Build short summary for LLM context (ordered by request)
        previews = []
        for cid in req_ids:
            c = chunks.get(cid)
            if c:
                previews.append(f"[{cid}] {c.description[:100]}...")
        summary = "\n".join(previews) if previews else "No chunks retrieved"

        found_ids = set(chunks.keys())
        missing_ids = [cid for cid in req_ids if cid not in found_ids]

        # Keep existing JSON shape; switch to model_dump() and include url
        return {
            "success": True,
            "retrieved_count": len(found_ids),
            "requested_count": len(req_ids),
            "chunks": {cid: c.model_dump() for cid, c in chunks.items()},
            "missing_ids": missing_ids,
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"Error retrieving chunks {req_ids}: {e}")
        return {"success": False, "error": f"Failed to retrieve chunks: {e}"}


@tool
def get_section_from_iteration(section_name: str, iteration: int) -> Dict[str, Any]:
    """
    Retrieve a specific section from a specific iteration for analysis or comparison.

    KEY CAPABILITY: Access historical versions to track changes and improvements.
    Essential for understanding revision history and feedback application.

    USAGE PATTERNS:
    - Review progress: get_section_from_iteration("Introduction", 1) vs current
    - Before improvement: Check section state before applying feedback
    - Quality assessment: Compare sections across multiple iterations
    - ADVANCED: Call multiple times to track section evolution:
      * iteration 0 (original) → iteration 1 (first revision) → iteration 2 (final)

    COMPARISON WORKFLOW:
    1. get_section_from_iteration("Section", iteration_1)
    2. get_section_from_iteration("Section", iteration_2)
    3. Compare the content using your reasoning to identify changes

    Args:
        section_name: Exact section name as it appears in article (case-sensitive)
        iteration: Iteration number (0=original, 1=first revision, etc.)

    Returns:
        Dictionary with:
        - section_name: Confirmed section name
        - iteration: Confirmed iteration number
        - content: Full section content from that iteration
        - success: Whether section was found
    """
    memory = ConfigContext.get_memory_instance()
    if not memory:
        return {"success": False, "error": "Memory not initialized"}

    try:
        section_content = memory.get_section_by_iteration(section_name, iteration)
        if section_content:
            return {
                "success": True,
                "section_name": section_name,
                "iteration": iteration,
                "content": section_content,
            }
        else:
            return {
                "success": False,
                "error": f"Section '{section_name}' not found in iteration {iteration}",
            }
    except Exception as e:
        logger.error(
            f"Error retrieving section {section_name} from iteration {iteration}: {e}"
        )
        return {"success": False, "error": f"Failed to retrieve section: {str(e)}"}


@tool
def get_current_iteration() -> Dict[str, Any]:
    """
    Get the current iteration number in the collaborative writing process.

    ITERATION AWARENESS: Understand where you are in the writing workflow.
    Critical for context about progress and next steps.

    USAGE PATTERNS:
    - Context awareness: Know if working on initial draft (iteration 0) or revisions
    - Progress tracking: Understanding how many revision cycles have occurred
    - Workflow decisions: Different strategies for early vs late iterations
    - Feedback context: When to expect or request feedback based on iteration

    ITERATION MEANINGS:
    - Iteration 0: Initial draft creation phase
    - Iteration 1+: Revision and improvement phases
    - Higher iterations: More refined content with applied feedback

    STRATEGIC APPLICATIONS:
    - Early iterations: Focus on content creation and structure
    - Later iterations: Focus on refinement and feedback application
    - Final iterations: Focus on polish and completion

    Returns:
        Dictionary with:
        - current_iteration: Current iteration number (0, 1, 2, etc.)
        - success: Whether iteration was successfully retrieved
        - message: Human-readable status of current iteration
    """
    memory = ConfigContext.get_memory_instance()
    if not memory:
        return {"success": False, "error": "Memory not initialized"}

    try:
        current_iteration = memory.get_current_iteration()
        return {
            "success": True,
            "current_iteration": current_iteration,
            "message": f"Currently on iteration {current_iteration}",
        }
    except Exception as e:
        logger.error(f"Error getting current iteration: {e}")
        return {"success": False, "error": f"Failed to get current iteration: {str(e)}"}


@tool
def get_feedback(
    feedback_id: str = "", section_name: str = "", only_pending: bool = False
) -> Dict[str, Any]:
    """
    Flexible feedback retrieval with multiple query modes for different workflow needs.

    QUERY MODES:
    1. By ID: get_feedback(feedback_id="fb_123") - Get specific feedback
    2. By section: get_feedback(section_name="Introduction") - All feedback for section
    3. Pending only: get_feedback(only_pending=True) - All unapplied feedback
    4. Section + pending: get_feedback(section_name="Methods", only_pending=True)

    WORKFLOW INTEGRATION:
    - Before writing: Check pending feedback to address in revisions
    - During revision: Get section-specific feedback to guide improvements
    - After changes: Retrieve feedback by ID to mark as applied
    - Quality control: Review all feedback to ensure comprehensive coverage

    STRATEGIC APPLICATIONS:
    - Revision planning: Use only_pending=True to prioritize remaining work
    - Section focus: Get all feedback for problematic sections
    - Feedback tracking: Use specific IDs to manage feedback lifecycle
    - Progress monitoring: Compare total vs pending feedback counts

    Args:
        feedback_id: Specific feedback ID for targeted retrieval (optional)
        section_name: Section name to get all related feedback (optional)
        only_pending: Filter to show only unapplied feedback (default: False)

    Returns:
        Dictionary with:
        - feedback_items: List of matching feedback entries
        - total_count: Number of matching feedback items
        - pending_count: Number of unapplied items in results
        - applied_count: Number of applied items in results
        - success: Whether query completed successfully
    """
    memory = ConfigContext.get_memory_instance()
    if not memory:
        return {"success": False, "error": "Memory not initialized"}

    try:
        if feedback_id:
            # Get specific feedback by ID
            feedback = memory.get_feedback_by_id(feedback_id)
            if feedback:
                return {"success": True, "feedback": feedback}
            else:
                return {
                    "success": False,
                    "error": f"Feedback with ID '{feedback_id}' not found",
                }

        elif section_name:
            # Get feedback for specific section
            feedback_list = memory.get_feedback_for_section(section_name)
            return {
                "success": True,
                "section_name": section_name,
                "feedback_count": len(feedback_list),
                "feedback": feedback_list,
            }

        elif only_pending:
            # Get pending feedback
            pending_feedback = memory.get_pending_feedback()
            return {
                "success": True,
                "pending_count": len(pending_feedback),
                "feedback": pending_feedback,
            }

        else:
            return {
                "success": False,
                "error": "Must specify either feedback_id, section_name, or only_pending=True",
            }

    except Exception as e:
        logger.error(f"Error retrieving feedback: {e}")
        return {"success": False, "error": f"Failed to retrieve feedback: {str(e)}"}


@tool
def mark_feedback_applied(feedback_id: str, notes: str = "") -> Dict[str, Any]:
    """
    Mark feedback as successfully applied with optional implementation notes.

    CRITICAL WORKFLOW STEP: Always call this after addressing feedback to:
    - Track which feedback has been implemented
    - Prevent re-applying the same feedback
    - Maintain clear revision history
    - Enable progress monitoring

    BEST PRACTICES:
    - Mark applied IMMEDIATELY after making changes
    - Add descriptive notes explaining how feedback was addressed
    - Use specific implementation details in notes
    - Call this for each piece of feedback individually

    USAGE PATTERNS:
    - Post-revision: mark_feedback_applied("fb_123", "Added three supporting examples")
    - Quality tracking: Note specific changes made in response to feedback
    - Progress updates: Mark applied to show completion of revision tasks
    - Audit trail: Provide notes for future reference and evaluation

    NOTES RECOMMENDATIONS:
    - Specific actions: "Added 2 paragraphs with statistical evidence"
    - Structural changes: "Reorganized section with clearer subheadings"
    - Content improvements: "Enhanced clarity by simplifying technical terms"

    Args:
        feedback_id: Exact feedback ID to mark as completed (required)
        notes: Detailed description of how feedback was implemented (recommended)

    Returns:
        Dictionary with:
        - feedback_id: Confirmed ID of processed feedback
        - status: New status (should be "applied")
        - notes: Implementation notes that were recorded
        - timestamp: When feedback was marked as applied
        - success: Whether operation completed successfully
    """
    memory = ConfigContext.get_memory_instance()
    if not memory:
        return {"success": False, "error": "Memory not initialized"}

    try:
        success = memory.mark_feedback_applied(feedback_id, notes)
        if success:
            return {
                "success": True,
                "feedback_id": feedback_id,
                "status": "applied",
                "notes": notes,
            }
        else:
            return {
                "success": False,
                "error": f"Feedback with ID '{feedback_id}' not found",
            }
    except Exception as e:
        logger.error(f"Error marking feedback {feedback_id} as applied: {e}")
        return {
            "success": False,
            "error": f"Failed to mark feedback as applied: {str(e)}",
        }


@tool
def mark_feedback_ignored(feedback_id: str, notes: str = "") -> Dict[str, Any]:
    """
    Mark feedback as ignored.

    Use this to mark feedback that you've decided not to address.

    Args:
        feedback_id: ID of the feedback to mark as ignored
        notes: Optional notes about why the feedback was ignored

    Returns:
        Dictionary confirming the feedback was marked as ignored
    """
    memory = ConfigContext.get_memory_instance()
    if not memory:
        return {"success": False, "error": "Memory not initialized"}

    try:
        success = memory.mark_feedback_ignored(feedback_id, notes)
        if success:
            return {
                "success": True,
                "feedback_id": feedback_id,
                "status": "ignored",
                "notes": notes,
            }
        else:
            return {
                "success": False,
                "error": f"Feedback with ID '{feedback_id}' not found",
            }
    except Exception as e:
        logger.error(f"Error marking feedback {feedback_id} as ignored: {e}")
        return {
            "success": False,
            "error": f"Failed to mark feedback as ignored: {str(e)}",
        }


@tool
def verify_claims_with_research(claims) -> Dict[str, Any]:
    """
    Comprehensive fact-checking tool: takes claims, gets research summary,
    selects relevant chunks, retrieves content, and returns verification results.

    This replaces inefficient search-based fact-checking with targeted approach
    using only the research chunks that the writer actually had access to.

    Args:
        claims: Either a comma-separated string or list of factual claims to verify

    Returns:
        Dictionary with verification results for each claim
    """
    import logging

    from src.config.config_context import ConfigContext

    logger = logging.getLogger(__name__)

    try:
        # Parse claims (support string or list inputs)
        if claims is None:
            return {
                "success": False,
                "error": "No claims provided for verification",
                "verifications": [],
            }

        if isinstance(claims, str):
            raw_claims = [part.strip() for part in claims.split(",") if part.strip()]
        elif isinstance(claims, list):
            raw_claims = [str(part).strip() for part in claims if str(part).strip()]
        else:
            raw_claims = [str(claims).strip()]

        claims_list = [claim for claim in raw_claims if claim]
        if not claims_list:
            return {
                "success": False,
                "error": "No valid claims found",
                "verifications": [],
            }

        # Get memory instance
        memory = ConfigContext.get_memory_instance()
        if not memory:
            return {
                "success": False,
                "error": "Memory not initialized",
                "verifications": [],
            }

        # Get research chunks summary from memory
        stored_chunks = {chunk.chunk_id: chunk for chunk in memory.get_stored_chunks()}
        if not stored_chunks:
            return {
                "success": False,
                "error": "No research chunks available for verification",
                "verifications": [],
            }

        # Select relevant chunks for each claim using simple keyword matching
        # (Could be enhanced with embedding similarity later)
        verifications = []

        for claim in claims_list:
            relevant_chunk_ids = []
            claim_lower = claim.lower()

            # Find chunks that might be relevant to this claim
            for chunk_id, chunk in stored_chunks.items():
                description = chunk.description.lower()
                source = chunk.source.lower()

                # Simple relevance scoring based on keyword overlap
                claim_words = set(claim_lower.split())
                desc_words = set(description.split())
                source_words = set(source.split())

                # Remove common words for better matching
                stop_words = {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                    "of",
                    "with",
                    "by",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "have",
                    "has",
                    "had",
                }
                claim_words = claim_words - stop_words
                desc_words = desc_words - stop_words
                source_words = source_words - stop_words

                # Calculate relevance score
                desc_overlap = len(claim_words & desc_words)
                source_overlap = len(claim_words & source_words)
                total_overlap = desc_overlap + source_overlap

                if total_overlap > 0:
                    relevant_chunk_ids.append((chunk_id, total_overlap))

            # Sort by relevance and take top 3 chunks
            relevant_chunk_ids.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [chunk_id for chunk_id, _ in relevant_chunk_ids[:3]]

            # Get full content for relevant chunks
            chunk_contents = []
            for chunk_id in top_chunks:
                chunk = stored_chunks.get(chunk_id)
                if not chunk:
                    continue
                chunk_contents.append(
                    {
                        "chunk_id": chunk_id,
                        "content": chunk.content[:500],  # Limit content length
                        "source": chunk.source,
                        "url": chunk.url,
                        "description": chunk.description,
                    }
                )

            # Create verification result
            verification = {
                "claim": claim,
                "relevant_chunks_found": len(chunk_contents),
                "chunk_contents": chunk_contents,
                "verified": len(chunk_contents)
                > 0,  # Simple verification - has supporting content
                "verification_confidence": (
                    "medium"
                    if len(chunk_contents) > 1
                    else "low" if len(chunk_contents) == 1 else "none"
                ),
            }

            verifications.append(verification)
            logger.debug(
                f"Verified claim '{claim[:50]}...' → {len(chunk_contents)} relevant chunks"
            )

        return {
            "success": True,
            "total_claims": len(claims_list),
            "research_chunks_available": len(stored_chunks),
            "verifications": verifications,
        }

    except Exception as e:
        logger.error(f"Error in verify_claims_with_research: {e}")
        return {
            "success": False,
            "error": f"Verification failed: {str(e)}",
            "verifications": [],
        }
