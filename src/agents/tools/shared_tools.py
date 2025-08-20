# src/agents/tools/shared_tools.py
"""
Shared real tools for both Writer and Reviewer agents.
These tools provide external capabilities that LLMs cannot perform alone.
"""

import logging
import numpy as np
import re
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional

from src.config.retrieval_config import RetrievalConfig
from src.retrieval.factory import create_retrieval_manager
from src.utils.data import SearchResult

logger = logging.getLogger(__name__)

# Global instances for efficiency
_embedding_model = None
_retrieval_managers = {}


def _get_embedding_model():
    """Get or create embedding model for semantic operations."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_retrieval_manager(rm_type: str = "wiki"):
    """Get or create retrieval manager using factory pattern."""
    global _retrieval_managers

    if rm_type not in _retrieval_managers:
        try:
            config = RetrievalConfig()
            config.retrieval_manager_type = rm_type
            _retrieval_managers[rm_type] = create_retrieval_manager(
                retrieval_config=config
            )
            logger.info(f"Created {rm_type} retrieval manager")
        except Exception as e:
            logger.error(f"Failed to create {rm_type} retrieval manager: {e}")
            # Fallback to wiki if other types fail
            if rm_type != "wiki":
                return _get_retrieval_manager("wiki")
            raise

    return _retrieval_managers[rm_type]


@tool
def search_and_retrieve(
    query: str, rm_type: str = "wiki", max_results: int = 5, purpose: str = "general"
) -> Dict[str, Any]:
    """
    Search external sources for information using configurable retrieval manager.

    Args:
        query: Search query string
        rm_type: Retrieval manager type ("wiki", "faiss_wiki", "bm25_wiki")
        max_results: Maximum number of results to return
        purpose: Purpose of search ("writing", "fact_checking", "verification")

    Returns:
        Dictionary with search results and metadata
    """
    try:
        rm = _get_retrieval_manager(rm_type)

        # Search using the retrieval manager
        passages = rm.search(
            query=query, max_results=max_results, format_type="rag", topic=query
        )

        # Convert to SearchResult objects
        search_results = []
        for i, passage in enumerate(passages):
            search_result = SearchResult(
                content=passage,
                source=f"{rm_type}_{i+1}",
                relevance_score=1.0 - (i * 0.1),
                metadata={"rm_type": rm_type, "purpose": purpose},
            )
            search_results.append(search_result)

        return {
            "query": query,
            "rm_type": rm_type,
            "purpose": purpose,
            "results_count": len(search_results),
            "results": [result.to_dict() for result in search_results],
            "summary": f"Found {len(search_results)} results for '{query}' using {rm_type}",
        }

    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        return {
            "query": query,
            "rm_type": rm_type,
            "purpose": purpose,
            "results_count": 0,
            "results": [],
            "error": str(e),
            "summary": f"Search failed for '{query}'",
        }


@tool
def organize_knowledge(
    topic: str,
    search_results_data: List[Dict[str, Any]],
    purpose: str = "writing",
    categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Organize search results into structured knowledge for different purposes.

    Args:
        topic: Main topic being organized
        search_results_data: List of search result dictionaries
        purpose: Organization purpose ("writing", "fact_checking", "verification")
        categories: Optional list of categories to organize by

    Returns:
        Dictionary with organized knowledge structure
    """
    try:
        # Convert to SearchResult objects
        search_results = [SearchResult(**data) for data in search_results_data]

        if not search_results:
            return {
                "topic": topic,
                "purpose": purpose,
                "categories": {},
                "summary": "No search results to organize",
            }

        # Define default categories based on purpose
        if categories is None:
            if purpose == "writing":
                categories = [
                    "Background_Information",
                    "Key_Concepts",
                    "Current_Developments",
                    "Examples_and_Cases",
                    "Technical_Details",
                ]
            elif purpose == "fact_checking":
                categories = [
                    "Verifiable_Facts",
                    "Statistical_Data",
                    "Contradictory_Information",
                    "Source_Citations",
                    "Uncertain_Claims",
                ]
            else:  # verification
                categories = [
                    "Supporting_Evidence",
                    "Contradicting_Evidence",
                    "Related_Context",
                    "Source_Reliability",
                ]

        # Organize results using semantic similarity
        embedding_model = _get_embedding_model()
        category_embeddings = embedding_model.encode(categories)

        organized_results = {category: [] for category in categories}
        unorganized_results = []

        for result in search_results:
            # Find best matching category
            content_embedding = embedding_model.encode([result.content])
            similarities = np.dot(content_embedding, category_embeddings.T)[0]
            best_category_idx = np.argmax(similarities)
            best_similarity = similarities[best_category_idx]

            # Only assign if similarity is above threshold
            if best_similarity > 0.3:
                best_category = categories[best_category_idx]
                organized_results[best_category].append(
                    {**result.to_dict(), "category_confidence": float(best_similarity)}
                )
            else:
                unorganized_results.append(result.to_dict())

        # Calculate coverage metrics
        total_results = len(search_results)
        organized_count = sum(len(results) for results in organized_results.values())
        coverage_score = organized_count / total_results if total_results > 0 else 0

        return {
            "topic": topic,
            "purpose": purpose,
            "total_results": total_results,
            "organized_count": organized_count,
            "coverage_score": coverage_score,
            "categories": organized_results,
            "unorganized": unorganized_results,
            "category_summary": {
                cat: len(results) for cat, results in organized_results.items()
            },
            "summary": f"Organized {organized_count}/{total_results} results into {len(categories)} categories",
        }

    except Exception as e:
        logger.error(f"Knowledge organization failed for topic '{topic}': {e}")
        return {
            "topic": topic,
            "purpose": purpose,
            "categories": {},
            "error": str(e),
            "summary": f"Organization failed for '{topic}'",
        }


@tool
def extract_claims(
    content: str, claim_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Extract factual claims from content using NLP processing.

    Args:
        content: Text content to analyze
        claim_types: Optional list of claim types to focus on

    Returns:
        Dictionary with extracted claims and metadata
    """
    try:
        if not content or not content.strip():
            return {
                "content_length": 0,
                "claims_found": 0,
                "claims": [],
                "summary": "No content provided for claim extraction",
            }

        # Define claim patterns based on linguistic markers
        if claim_types is None:
            claim_types = ["factual", "statistical", "definitional", "causal"]

        sentences = re.split(r"(?<=[.!?])\s+", content.strip())
        extracted_claims = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue

            claim_info = {
                "text": sentence,
                "type": "unknown",
                "confidence": 0.0,
                "indicators": [],
            }

            # Factual claims (is/are statements)
            if re.search(
                r"\b(is|are|was|were|has|have|contains?|includes?)\b",
                sentence,
                re.IGNORECASE,
            ):
                claim_info["type"] = "factual"
                claim_info["confidence"] = 0.7
                claim_info["indicators"].append("factual_verb")

            # Statistical claims (numbers, percentages)
            if re.search(
                r"\b\d+([.,]\d+)?%?\b|\b(percent|million|billion|thousand)\b",
                sentence,
                re.IGNORECASE,
            ):
                claim_info["type"] = "statistical"
                claim_info["confidence"] = 0.8
                claim_info["indicators"].append("numerical_data")

            # Definitional claims
            if re.search(
                r"\b(defined as|refers to|means|called)\b", sentence, re.IGNORECASE
            ):
                claim_info["type"] = "definitional"
                claim_info["confidence"] = 0.6
                claim_info["indicators"].append("definition_marker")

            # Causal claims
            if re.search(
                r"\b(because|due to|causes?|results? in|leads? to)\b",
                sentence,
                re.IGNORECASE,
            ):
                claim_info["type"] = "causal"
                claim_info["confidence"] = 0.5
                claim_info["indicators"].append("causal_marker")

            # Only include claims with some confidence
            if claim_info["confidence"] > 0.0:
                extracted_claims.append(claim_info)

        # Sort by confidence
        extracted_claims.sort(key=lambda x: x["confidence"], reverse=True)

        # Limit to top claims to avoid overwhelming
        extracted_claims = extracted_claims[:15]

        return {
            "content_length": len(content),
            "sentences_analyzed": len(sentences),
            "claims_found": len(extracted_claims),
            "claims": extracted_claims,
            "claim_types_found": list(set(claim["type"] for claim in extracted_claims)),
            "average_confidence": (
                np.mean([claim["confidence"] for claim in extracted_claims])
                if extracted_claims
                else 0.0
            ),
            "summary": f"Extracted {len(extracted_claims)} claims from {len(sentences)} sentences",
        }

    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        return {
            "content_length": len(content) if content else 0,
            "claims_found": 0,
            "claims": [],
            "error": str(e),
            "summary": "Claim extraction failed",
        }


@tool
def verify_claims_against_sources(
    claims: List[Dict[str, Any]],
    source_results: List[Dict[str, Any]],
    verification_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Cross-reference claims against source material using semantic similarity.

    Args:
        claims: List of claim dictionaries from extract_claims
        source_results: List of search result dictionaries
        verification_threshold: Minimum similarity for claim verification

    Returns:
        Dictionary with verification results
    """
    try:
        if not claims or not source_results:
            return {
                "total_claims": len(claims),
                "total_sources": len(source_results),
                "verified_claims": [],
                "unverified_claims": claims,
                "verification_summary": {},
                "summary": "No claims or sources provided for verification",
            }

        embedding_model = _get_embedding_model()

        # Extract source texts
        source_texts = []
        for source in source_results:
            if isinstance(source, dict) and "content" in source:
                source_texts.append(source["content"])
            elif isinstance(source, str):
                source_texts.append(source)

        if not source_texts:
            return {
                "total_claims": len(claims),
                "total_sources": len(source_results),
                "verified_claims": [],
                "unverified_claims": claims,
                "summary": "No valid source texts found for verification",
            }

        # Encode all source texts
        source_embeddings = embedding_model.encode(source_texts)

        verified_claims = []
        unverified_claims = []

        for claim in claims:
            claim_text = claim.get("text", "")
            if not claim_text:
                unverified_claims.append(claim)
                continue

            # Encode claim
            claim_embedding = embedding_model.encode([claim_text])

            # Find best matching source
            similarities = np.dot(claim_embedding, source_embeddings.T)[0]
            best_source_idx = np.argmax(similarities)
            best_similarity = similarities[best_source_idx]

            verification_result = {
                **claim,
                "verification_score": float(best_similarity),
                "best_source_idx": int(best_source_idx),
                "best_source_content": (
                    source_texts[best_source_idx][:200] + "..."
                    if len(source_texts[best_source_idx]) > 200
                    else source_texts[best_source_idx]
                ),
                "verified": bool(best_similarity >= verification_threshold),
            }

            if best_similarity >= verification_threshold:
                verified_claims.append(verification_result)
            else:
                unverified_claims.append(verification_result)

        # Calculate verification metrics
        verification_rate = len(verified_claims) / len(claims) if claims else 0
        avg_verification_score = (
            np.mean([claim["verification_score"] for claim in verified_claims])
            if verified_claims
            else 0
        )

        return {
            "total_claims": len(claims),
            "total_sources": len(source_results),
            "verified_claims": verified_claims,
            "unverified_claims": unverified_claims,
            "verification_rate": verification_rate,
            "average_verification_score": avg_verification_score,
            "verification_threshold": verification_threshold,
            "verification_summary": {
                "high_confidence": len(
                    [c for c in verified_claims if c["verification_score"] > 0.8]
                ),
                "medium_confidence": len(
                    [
                        c
                        for c in verified_claims
                        if 0.6 <= c["verification_score"] <= 0.8
                    ]
                ),
                "low_confidence": len(
                    [c for c in unverified_claims if c["verification_score"] > 0.4]
                ),
                "no_evidence": len(
                    [c for c in unverified_claims if c["verification_score"] <= 0.4]
                ),
            },
            "summary": f"Verified {len(verified_claims)}/{len(claims)} claims against {len(source_results)} sources",
        }

    except Exception as e:
        logger.error(f"Claim verification failed: {e}")
        return {
            "total_claims": len(claims),
            "total_sources": len(source_results),
            "verified_claims": [],
            "unverified_claims": claims,
            "error": str(e),
            "summary": "Claim verification failed",
        }
