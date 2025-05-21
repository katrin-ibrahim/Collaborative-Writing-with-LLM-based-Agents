import json
import re
import requests
from typing import Any, Dict, Optional


class ContentEvaluator:
    """Evaluator for article content quality"""

    def __init__(self, api_url=None, api_key=None):
        # Use same API setup as writer agent
        self.api_url = (
            api_url or "https://router.huggingface.co/novita/v3/openai/chat/completions"
        )
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _call_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call API with prompt"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "model": "deepseek/deepseek-v3-0324",
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        output = response.json()

        # Handle OpenAI-compatible response format
        if isinstance(output, dict) and "choices" in output:
            choices = output.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict) and "content" in message:
                    return message["content"].strip()

        return ""

    def evaluate_article(self, content: str, topic: str) -> Dict[str, Any]:
        """Evaluate article content"""
        # Extract metrics
        metrics = self._extract_metrics(content)

        # LLM evaluation
        llm_eval = self._llm_evaluation(content, topic)

        # Structure evaluation
        structure_eval = self._evaluate_structure(content)

        # Combine all evaluations
        evaluation = {
            "metrics": metrics,
            "llm_evaluation": llm_eval,
            "structure_evaluation": structure_eval,
            "overall_score": llm_eval.get("overall_score", 0),
        }

        return evaluation

    def _extract_metrics(self, content: str) -> Dict[str, Any]:
        """Extract basic metrics from content"""
        # Get word count
        words = re.findall(r"\b\w+\b", content)
        word_count = len(words)

        # Get section count
        sections = re.findall(r"^##\s+.+$", content, re.MULTILINE)
        section_count = len(sections)

        # Get paragraph count
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        return {
            "word_count": word_count,
            "section_count": section_count,
            "paragraph_count": paragraph_count,
            "avg_words_per_section": word_count / max(section_count, 1),
        }

    def _llm_evaluation(self, content: str, topic: str) -> Dict[str, Any]:
        """Use LLM to evaluate content quality"""
        system_prompt = "You are an expert content evaluator providing objective assessments of article quality."

        prompt = (
            f"Evaluate this article about '{topic}':\n\n"
            f"```\n{content}\n```\n\n"
            f"Please rate on a scale of 1-10 for each criterion:\n"
            f"1) Informativeness: How well does it convey valuable information?\n"
            f"2) Coherence: How well-structured and logical is the content?\n"
            f"3) Clarity: How clear and understandable is the writing?\n"
            f"4) Engagement: How engaging and interesting is the content?\n"
            f"5) Accuracy: How accurate does the information appear to be?\n"
            f"6) Overall: What is the overall quality?\n\n"
            f"For each criterion, provide a score and brief explanation. Then calculate the overall score "
            f"as the average of the five individual scores.\n\n"
            f"Respond in JSON format with these keys: informativeness, coherence, clarity, engagement, accuracy, "
            f"overall_score, and summary."
        )

        response = self._call_api(prompt, system_prompt)

        try:
            evaluation = json.loads(response)
            return evaluation
        except json.JSONDecodeError:
            # Extract JSON if embedded in text
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    evaluation = json.loads(match.group(0))
                    return evaluation
                except:
                    pass

        # Fallback for parsing failures
        return {
            "informativeness": 5,
            "coherence": 5,
            "clarity": 5,
            "engagement": 5,
            "accuracy": 5,
            "overall_score": 5,
            "summary": "Failed to parse evaluation",
        }

    def _evaluate_structure(self, content: str) -> Dict[str, Any]:
        """Evaluate the structure and organization of the content"""
        # Extract section headings
        headings = re.findall(r"^##\s+(.+)$", content, re.MULTILINE)

        if not headings:
            return {"error": "No section headings found"}

        system_prompt = (
            "You are an expert evaluator of document structure and organization."
        )

        prompt = (
            f"Evaluate the structural organization based on these section headings:\n\n"
            f"{', '.join(headings)}\n\n"
            f"Rate on a scale of 1-10 for each criterion:\n"
            f"1) Logical flow: Do the sections follow a logical progression?\n"
            f"2) Comprehensiveness: Do the headings cover the important aspects of the topic?\n"
            f"3) Balance: Are the sections appropriately balanced in scope?\n"
            f"4) Clarity: Are the headings clear and informative?\n\n"
            f"Provide ratings as JSON with these keys: logical_flow, comprehensiveness, balance, clarity, "
            f"overall_structure_score, and suggestions_for_improvement."
        )

        response = self._call_api(prompt, system_prompt)

        try:
            return json.loads(response)
        except:
            # Simplified fallback
            return {
                "logical_flow": 5,
                "comprehensiveness": 5,
                "balance": 5,
                "clarity": 5,
                "overall_structure_score": 5,
                "suggestions_for_improvement": "Could not parse evaluation.",
            }
