# src/agents/reviewer/reviewer_tools.py
from langchain_core.tools import tool
from typing import Any, Dict, List

from src.agents.reviewer.data_models import Claim
from src.agents.tools.knowledge_toolkit import extract_claims_from_content
from src.agents.tools.search_toolkit import create_search_tool


@tool
def extract_verifiable_claims(content: str, max_claims: int = 10) -> Dict[str, Any]:
    """
    Extract verifiable claims from article content for fact-checking.

    Args:
        content: Article content to analyze
        max_claims: Maximum number of claims to extract

    Returns:
        Dictionary with extracted claims and metadata
    """
    # Use existing extract_claims_from_content tool
    result = extract_claims_from_content.invoke({"content": content})

    # Convert to Claim objects with enhanced metadata
    claims = []
    for i, claim_text in enumerate(result["extracted_claims"][:max_claims]):
        claim = Claim(
            text=claim_text,
            section="main",  # TODO: Improve section detection
            line_number=None,
            claim_type="factual",
            confidence=0.8,  # Default confidence
        )
        claims.append(claim)

    return {
        "content_length": result["content_length"],
        "claims_extracted": len(claims),
        "claims": [
            {
                "text": claim.text,
                "section": claim.section,
                "claim_type": claim.claim_type,
                "confidence": claim.confidence,
            }
            for claim in claims
        ],
        "summary": f"Extracted {len(claims)} verifiable claims for fact-checking",
    }


def create_reviewer_tools(retrieval_config=None):
    """Create reviewer tools with specific retrieval configuration."""

    # Create search tool with the provided config
    search_tool = create_search_tool(retrieval_config)

    @tool
    def fact_check_claim(claim_text: str, context: str = "") -> Dict[str, Any]:
        """
        Verify a specific claim against available sources.

        Args:
            claim_text: The claim to verify
            context: Additional context for the search

        Returns:
            Dictionary with fact-check results and evidence
        """
        # Search for evidence using configured search tool
        search_query = f"{claim_text} {context}".strip()
        search_result = search_tool.invoke(
            {"query": search_query, "wiki_results": 3, "web_results": 2}
        )

        # Analyze evidence (simplified heuristic approach)
        supporting_evidence = []
        contradicting_evidence = []
        sources = []

        for result_data in search_result["results"]:
            content = result_data["content"].lower()
            claim_lower = claim_text.lower()

            # Simple keyword matching for evidence classification
            if any(word in content for word in claim_lower.split()[:3]):
                supporting_evidence.append(result_data["content"][:200] + "...")
                sources.append(result_data["source"])

        # Calculate accuracy score based on evidence
        accuracy_score = min(1.0, len(supporting_evidence) * 0.3)
        verification_status = (
            "verified" if accuracy_score > 0.6 else "needs_verification"
        )

        return {
            "claim": claim_text,
            "accuracy_score": accuracy_score,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "verification_status": verification_status,
            "sources": sources,
            "evidence_count": len(supporting_evidence),
            "summary": f"Fact-checked claim with {len(supporting_evidence)} supporting sources",
        }

    return fact_check_claim


# Default reviewer tools for backward compatibility
fact_check_claim = create_reviewer_tools()


@tool
def analyze_article_structure(article_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze article organization and structural quality.

    Args:
        article_dict: Article data as dictionary

    Returns:
        Dictionary with structure analysis results
    """
    title = article_dict.get("title", "")
    content = article_dict.get("content", "")
    sections = article_dict.get("sections", {})

    # Basic structure analysis
    outline_score = 0.8 if len(sections) > 2 else 0.5
    transition_score = 0.7  # Default - would need more sophisticated analysis
    coherence_score = 0.8 if len(content) > 500 else 0.6
    completeness_score = min(1.0, len(content) / 1000)  # Based on content length

    issues = []
    recommendations = []

    # Check for common structural issues
    if len(sections) < 3:
        issues.append("Article has too few sections for comprehensive coverage")
        recommendations.append("Consider adding more detailed sections")

    if len(content) < 300:
        issues.append("Article content is too brief")
        recommendations.append("Expand content with more detailed information")

    if not title:
        issues.append("Article lacks a clear title")
        recommendations.append("Add a descriptive title")

    return {
        "outline_score": outline_score,
        "transition_score": transition_score,
        "coherence_score": coherence_score,
        "completeness_score": completeness_score,
        "overall_structure_score": (
            outline_score + transition_score + coherence_score + completeness_score
        )
        / 4,
        "issues": issues,
        "recommendations": recommendations,
        "sections_count": len(sections),
        "content_length": len(content),
        "summary": f"Analyzed article structure with {len(issues)} issues identified",
    }


@tool
def generate_structured_feedback(
    claims_results: List[Dict[str, Any]],
    structure_results: Dict[str, Any],
    article_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate categorized feedback with severity levels.

    Args:
        claims_results: Results from fact-checking claims
        structure_results: Results from structure analysis
        article_dict: Original article data

    Returns:
        Dictionary with structured feedback and recommendations
    """
    from src.agents.reviewer.data_models import FeedbackCategory, FeedbackItem, Severity

    issues = []
    recommendations = []
    category_scores = {}

    # Process fact-checking results
    factual_issues = 0
    total_claims = len(claims_results)

    for claim_result in claims_results:
        if claim_result["accuracy_score"] < 0.5:
            issues.append(
                FeedbackItem(
                    category=FeedbackCategory.FACTUAL,
                    severity=Severity.MAJOR,
                    description=f"Claim needs verification: {claim_result['claim'][:100]}...",
                    suggestion="Verify this claim with additional sources",
                    context=f"Accuracy score: {claim_result['accuracy_score']:.2f}",
                )
            )
            factual_issues += 1

    category_scores[FeedbackCategory.FACTUAL] = 1.0 - (
        factual_issues / max(1, total_claims)
    )

    # Process structure analysis
    structure_score = structure_results["overall_structure_score"]
    category_scores[FeedbackCategory.STRUCTURAL] = structure_score

    for issue in structure_results["issues"]:
        issues.append(
            FeedbackItem(
                category=FeedbackCategory.STRUCTURAL,
                severity=Severity.MINOR,
                description=issue,
                suggestion="Improve article organization",
            )
        )

    # Add structure recommendations
    recommendations.extend(structure_results["recommendations"])

    # Calculate overall score
    overall_score = sum(category_scores.values()) / len(category_scores)

    # Generate summary
    summary = f"Review completed: {len(issues)} issues found, overall score {overall_score:.2f}"

    return {
        "overall_score": overall_score,
        "category_scores": {cat.value: score for cat, score in category_scores.items()},
        "issues": [
            {
                "category": item.category.value,
                "severity": item.severity.value,
                "description": item.description,
                "location": item.location,
                "suggestion": item.suggestion,
                "context": item.context,
            }
            for item in issues
        ],
        "recommendations": recommendations,
        "summary": summary,
        "metadata": {
            "claims_checked": total_claims,
            "factual_issues": factual_issues,
            "structure_score": structure_score,
            "review_timestamp": "2025-01-08",  # Would use actual timestamp
        },
    }
