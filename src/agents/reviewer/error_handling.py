# src/agents/reviewer/error_handling.py
"""
Enhanced error handling and recovery mechanisms for ReviewerAgent.
"""

import time
from functools import wraps

import logging
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3, backoff_factor: float = 1.0, exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exceptions to catch and retry
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = backoff_factor * (2**attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )
                        break

            raise last_exception

        return wrapper

    return decorator


def safe_execute(
    operation: Callable, fallback_value: Any = None, error_context: str = "operation"
) -> Any:
    """
    Safely execute an operation with fallback value on error.

    Args:
        operation: Function to execute
        fallback_value: Value to return on error
        error_context: Context description for logging

    Returns:
        Operation result or fallback value
    """
    try:
        return operation()
    except Exception as e:
        logger.error(f"Safe execution failed for {error_context}: {e}")
        return fallback_value


class ErrorRecoveryManager:
    """Manages error recovery strategies for ReviewerAgent operations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_counts = {}
        self.recovery_strategies = {
            "claim_extraction": self._recover_claim_extraction,
            "fact_checking": self._recover_fact_checking,
            "structure_analysis": self._recover_structure_analysis,
            "feedback_generation": self._recover_feedback_generation,
        }

    def handle_error(
        self, error: Exception, operation: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle errors with appropriate recovery strategy.

        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            context: Context information for recovery

        Returns:
            Recovery result or fallback data
        """
        self.error_counts[operation] = self.error_counts.get(operation, 0) + 1

        self.logger.error(f"Error in {operation}: {error}")
        self.logger.info(
            f"Attempting recovery for {operation} (attempt #{self.error_counts[operation]})"
        )

        if operation in self.recovery_strategies:
            try:
                return self.recovery_strategies[operation](error, context)
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {operation}: {recovery_error}")
                return self._get_minimal_fallback(operation, context)
        else:
            return self._get_minimal_fallback(operation, context)

    def _recover_claim_extraction(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recovery strategy for claim extraction failures."""
        content = context.get("content", "")

        # Try simple sentence-based extraction as fallback
        try:
            sentences = [s.strip() for s in content.split(".") if s.strip()]

            # Simple heuristic: look for sentences with numbers, dates, or specific claims
            potential_claims = []
            for sentence in sentences[:10]:  # Limit to first 10 sentences
                if any(
                    indicator in sentence.lower()
                    for indicator in [
                        "composed",
                        "wrote",
                        "born",
                        "died",
                        "created",
                        "invented",
                        "discovered",
                    ]
                ):
                    potential_claims.append(
                        {
                            "text": sentence,
                            "section": "main",
                            "claim_type": "factual",
                            "confidence": 0.5,  # Lower confidence for fallback
                        }
                    )

            return {
                "claims": potential_claims[:5],  # Limit to 5 claims
                "recovery_method": "simple_heuristic",
                "original_error": str(error),
            }
        except Exception:
            return {
                "claims": [],
                "recovery_method": "failed",
                "original_error": str(error),
            }

    def _recover_fact_checking(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recovery strategy for fact-checking failures."""
        claim = context.get("claim", "")

        return {
            "claim": claim,
            "accuracy_score": 0.5,  # Neutral score when unable to verify
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "verification_status": "unverifiable",
            "sources": [],
            "evidence_count": 0,
            "confidence": 0.0,
            "reasoning": f"Fact-checking failed: {str(error)}",
            "recovery_method": "neutral_fallback",
            "original_error": str(error),
        }

    def _recover_structure_analysis(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recovery strategy for structure analysis failures."""
        article = context.get("article", {})
        content_length = len(article.get("content", ""))
        sections_count = len(article.get("sections", {}))

        # Calculate basic fallback scores based on simple metrics
        length_score = min(
            1.0, content_length / 1000
        )  # 1000 chars = perfect length score
        section_score = min(
            1.0, sections_count / 5
        )  # 5 sections = perfect section score
        fallback_score = (length_score + section_score) / 2

        return {
            "outline_score": fallback_score,
            "transition_score": fallback_score,
            "coherence_score": fallback_score,
            "completeness_score": fallback_score,
            "paragraph_organization_score": fallback_score,
            "overall_structure_score": fallback_score,
            "issues": [f"Structure analysis failed: {str(error)}"],
            "recommendations": ["Manual structural review recommended"],
            "strengths": [],
            "overall_assessment": f"Fallback analysis (score: {fallback_score:.2f})",
            "sections_count": sections_count,
            "content_length": content_length,
            "recovery_method": "basic_metrics",
            "original_error": str(error),
        }

    def _recover_feedback_generation(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recovery strategy for feedback generation failures."""
        fact_check_results = context.get("fact_check_results", [])
        structure_analysis = context.get("structure_analysis", {})

        # Calculate basic scores from available data
        factual_score = 0.5
        if fact_check_results:
            verified_count = sum(
                1
                for r in fact_check_results
                if r.get("verification_status") == "verified"
            )
            factual_score = verified_count / len(fact_check_results)

        structural_score = structure_analysis.get("overall_structure_score", 0.5)
        overall_score = (factual_score + structural_score) / 2

        return {
            "overall_score": overall_score,
            "category_scores": {
                "factual": factual_score,
                "structural": structural_score,
                "clarity": 0.5,
                "completeness": 0.5,
                "style": 0.5,
            },
            "issues": [
                {
                    "category": "completeness",
                    "severity": "major",
                    "description": f"Review process encountered an error: {str(error)}",
                    "location": "system",
                    "suggestion": "Manual review recommended",
                    "context": "Automated review failed",
                }
            ],
            "recommendations": [
                "Manual review recommended due to processing error",
                "Verify all factual claims independently",
                "Review article structure and organization",
            ],
            "summary": f"Partial review completed with errors (score: {overall_score:.2f})",
            "recovery_method": "basic_calculation",
            "original_error": str(error),
        }

    def _get_minimal_fallback(
        self, operation: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get minimal fallback data for unknown operations."""
        return {
            "error": f"Unknown operation: {operation}",
            "recovery_method": "minimal_fallback",
            "context": str(context),
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_breakdown": dict(self.error_counts),
            "most_problematic_operation": (
                max(self.error_counts.items(), key=lambda x: x[1])[0]
                if self.error_counts
                else None
            ),
        }


def validate_reviewer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set defaults for reviewer configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Validated configuration with defaults
    """
    defaults = {
        "reviewer.max_claims_per_article": 10,
        "reviewer.fact_check_timeout": 30,
        "reviewer.min_claim_confidence": 0.7,
        "reviewer.enable_structure_analysis": True,
        "reviewer.feedback_detail_level": "detailed",
        "reviewer.max_retries": 3,
        "reviewer.enable_error_recovery": True,
        "reviewer.severity_thresholds": {"critical": 0.9, "major": 0.7, "minor": 0.5},
    }

    # Apply defaults for missing keys
    validated_config = dict(config)
    for key, default_value in defaults.items():
        if key not in validated_config:
            validated_config[key] = default_value

    # Validate ranges
    validated_config["reviewer.max_claims_per_article"] = max(
        1, min(50, validated_config["reviewer.max_claims_per_article"])
    )
    validated_config["reviewer.fact_check_timeout"] = max(
        5, min(300, validated_config["reviewer.fact_check_timeout"])
    )
    validated_config["reviewer.min_claim_confidence"] = max(
        0.0, min(1.0, validated_config["reviewer.min_claim_confidence"])
    )

    return validated_config
