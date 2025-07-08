from typing import Any, Dict, List

from utils.data_models import SearchResult


class EvaluationToolkit:
    """
    Evaluation tools that can be used by any agent.

    Writer agents use these for self-assessment.
    Future: Reviewer agents will use these extensively for content evaluation.
    """

    def assess_content_gaps(
        self, topic: str, current_sections: List[str], available_info: str, api_client
    ) -> Dict[str, Any]:
        """Assess what content gaps exist."""
        assessment_prompt = f"""
        Assess the content coverage for an article about "{topic}".

        Current sections written: {current_sections}
        Available information length: {len(available_info)} characters

        Evaluate:
        1. What important aspects of {topic} are missing?
        2. Is the available information sufficient?
        3. What specific topics should be researched more?

        Respond in JSON format:
        {{
            "missing_topics": ["topic1", "topic2"],
            "information_sufficient": true/false,
            "suggested_searches": ["search1", "search2"],
            "overall_completeness": 0.0-1.0
        }}
        """

        response = api_client.call_api(assessment_prompt)
        return self._parse_json_response(
            response,
            {
                "missing_topics": [],
                "information_sufficient": len(available_info) > 1000,
                "suggested_searches": [f"{topic} overview", f"{topic} examples"],
                "overall_completeness": 0.5,
            },
        )

    def evaluate_section_quality(
        self, section_content: str, section_title: str, api_client
    ) -> Dict[str, float]:
        """
        Evaluate the quality of a written section.
        Future: Core tool for reviewer agents.
        """
        evaluation_prompt = f"""
        Evaluate this section titled "{section_title}":

        {section_content}

        Rate each aspect from 0.0 to 1.0:
        1. Clarity and readability
        2. Factual depth and detail
        3. Logical organization
        4. Relevance to section title

        Respond in JSON format:
        {{
            "clarity": 0.8,
            "depth": 0.7,
            "organization": 0.9,
            "relevance": 0.8
        }}
        """

        response = api_client.call_api(evaluation_prompt)
        return self._parse_json_response(
            response,
            {"clarity": 0.7, "depth": 0.6, "organization": 0.7, "relevance": 0.8},
        )

    def check_factual_accuracy(
        self, content: str, verification_results: List[SearchResult], api_client
    ) -> Dict[str, Any]:
        """
        Check factual accuracy of content against verification sources.
        Future: Primary tool for reviewer fact-checking agents.
        """
        verification_context = "\n".join(
            [
                f"Source: {r.source}\n{r.content[:200]}..."
                for r in verification_results[:3]
            ]
        )

        accuracy_prompt = f"""
        Check the factual accuracy of this content:

        CONTENT TO CHECK:
        {content}

        VERIFICATION SOURCES:
        {verification_context}

        Evaluate:
        1. Are there any factual errors or unsupported claims?
        2. Does the content align with the verification sources?
        3. What claims need additional verification?

        Respond in JSON format:
        {{
            "accuracy_score": 0.0-1.0,
            "unsupported_claims": ["claim1", "claim2"],
            "factual_errors": ["error1", "error2"],
            "needs_verification": ["claim1", "claim2"]
        }}
        """

        response = api_client.call_api(accuracy_prompt)
        return self._parse_json_response(
            response,
            {
                "accuracy_score": 0.8,
                "unsupported_claims": [],
                "factual_errors": [],
                "needs_verification": [],
            },
        )

    def _parse_json_response(
        self, response: str, fallback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse JSON from API response with fallback."""
        try:
            import json
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return fallback
