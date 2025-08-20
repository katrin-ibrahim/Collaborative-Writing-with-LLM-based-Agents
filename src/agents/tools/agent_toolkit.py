# src/agents/tools/agent_toolkit.py
"""
Updated AgentToolkit using only real tools that provide external capabilities.
Removes fake ContentToolkit and fixes search_toolkit to use factory pattern.
"""

from typing import Any, Dict

from src.agents.tools.shared_tools import (
    extract_claims,
    organize_knowledge,
    search_and_retrieve,
    verify_claims_against_sources,
)


class AgentToolkit:
    """
    Simplified toolkit providing real tools for both Writer and Reviewer agents.
    Uses shared tools that provide genuine external capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Configuration for tool usage
        self.rm_type = config.get("retrieval_manager_type", "wiki")
        self.max_search_results = config.get("max_search_results", 5)
        self.verification_threshold = config.get("verification_threshold", 0.6)

        # Store tool references for easy access
        self.search_and_retrieve = search_and_retrieve
        self.organize_knowledge = organize_knowledge
        self.extract_claims = extract_claims
        self.verify_claims_against_sources = verify_claims_against_sources

    def search_for_content(
        self, query: str, purpose: str = "writing"
    ) -> Dict[str, Any]:
        """Convenience method for content-focused searches"""
        return self.search_and_retrieve.invoke(
            {
                "query": query,
                "rm_type": self.rm_type,
                "max_results": self.max_search_results,
                "purpose": purpose,
            }
        )

    def search_for_verification(self, claim: str) -> Dict[str, Any]:
        """Convenience method for fact-checking searches"""
        return self.search_and_retrieve.invoke(
            {
                "query": claim,
                "rm_type": self.rm_type,
                "max_results": 3,  # Fewer results for verification
                "purpose": "fact_checking",
            }
        )

    def organize_for_writing(self, topic: str, search_results: list) -> Dict[str, Any]:
        """Convenience method for organizing content for writing"""
        return self.organize_knowledge.invoke(
            {
                "topic": topic,
                "search_results_data": search_results,
                "purpose": "writing",
                "categories": None,  # Use default writing categories
            }
        )

    def organize_for_fact_checking(
        self, topic: str, evidence_results: list
    ) -> Dict[str, Any]:
        """Convenience method for organizing evidence for fact-checking"""
        return self.organize_knowledge.invoke(
            {
                "topic": topic,
                "search_results_data": evidence_results,
                "purpose": "fact_checking",
                "categories": [
                    "Supporting_Evidence",
                    "Contradictory_Evidence",
                    "Statistical_Data",
                    "Source_Citations",
                    "Uncertain_Information",
                ],
            }
        )

    def extract_content_claims(
        self, content: str, focus_types: list = None
    ) -> Dict[str, Any]:
        """Convenience method for extracting claims from content"""
        if focus_types is None:
            focus_types = ["factual", "statistical"]

        return self.extract_claims.invoke(
            {"content": content, "claim_types": focus_types}
        )

    def verify_claims(self, claims: list, sources: list) -> Dict[str, Any]:
        """Convenience method for verifying claims against sources"""
        return self.verify_claims_against_sources.invoke(
            {
                "claims": claims,
                "source_results": sources,
                "verification_threshold": self.verification_threshold,
            }
        )

    def get_available_tools(self) -> list:
        """Return list of available tools for LangGraph integration"""
        return [
            self.search_and_retrieve,
            self.organize_knowledge,
            self.extract_claims,
            self.verify_claims_against_sources,
        ]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Return descriptions of available tools for agent planning"""
        return {
            "search_and_retrieve": "Search external sources for information using configurable retrieval managers",
            "organize_knowledge": "Organize search results into structured categories for different purposes",
            "extract_claims": "Extract factual claims from text content using NLP processing",
            "verify_claims_against_sources": "Cross-reference claims against source material for fact-checking",
        }


# ============================================================================
# REMOVED COMPONENTS
# ============================================================================

"""
REMOVED: ContentToolkit - Was a fake tool that just created prompts

The ContentToolkit provided no external capabilities and was just wrapper
around LLM prompting. Its functions are now handled directly by agents:

- generate_outline() -> Agent uses LLM reasoning directly
- generate_section_content() -> Agent uses LLM with organized knowledge
- OutlineGenerator -> Replaced with direct LLM calls

This elimination follows the principle: Tools = External capabilities only
"""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_writer_usage():
    """Example of how Writer agent uses the real tools"""

    # Initialize toolkit
    config = {"retrieval_manager_type": "wiki", "max_search_results": 5}
    toolkit = AgentToolkit(config)

    # 1. Search for content (external capability)
    search_result = toolkit.search_for_content(
        query="quantum computing applications", purpose="writing"
    )

    # 2. Organize knowledge (computational processing)
    organized = toolkit.organize_for_writing(
        topic="Quantum Computing", search_results=search_result["results"]
    )

    # 3. Generate outline (LLM reasoning - no tool needed)
    # outline = llm.call("Create outline using: " + str(organized))

    # 4. Write content (LLM generation - no tool needed)
    # content = llm.call("Write section using: " + str(organized))

    # 5. Self-validate content (NLP processing)
    claims = toolkit.extract_content_claims(content="sample content")

    return {
        "search_results": len(search_result["results"]),
        "organized_categories": len(organized["categories"]),
        "extracted_claims": len(claims["claims"]),
    }


def example_reviewer_usage():
    """Example of how Reviewer agent uses the real tools"""

    # Initialize toolkit
    config = {"retrieval_manager_type": "wiki", "verification_threshold": 0.7}
    toolkit = AgentToolkit(config)

    # 1. Extract claims from article (NLP processing)
    article_content = (
        "Quantum computers are 1000 times faster than classical computers."
    )
    claims = toolkit.extract_content_claims(
        content=article_content, focus_types=["factual", "statistical", "causal"]
    )

    # 2. Search for verification evidence (external capability)
    evidence_results = []
    for claim in claims["claims"][:3]:  # Limit for efficiency
        evidence = toolkit.search_for_verification(claim["text"])
        evidence_results.extend(evidence["results"])

    # 3. Organize evidence (computational processing)
    organized_evidence = toolkit.organize_for_fact_checking(
        topic="Quantum Computing Performance", evidence_results=evidence_results
    )

    # 4. Verify claims against sources (computational matching)
    verification = toolkit.verify_claims(
        claims=claims["claims"], sources=evidence_results
    )

    # 5. Generate review (LLM reasoning - no tool needed)
    # review = llm.call("Create review based on: " + str(verification))

    return {
        "claims_extracted": len(claims["claims"]),
        "evidence_pieces": len(evidence_results),
        "verification_rate": verification["verification_rate"],
        "verified_claims": len(verification["verified_claims"]),
    }
