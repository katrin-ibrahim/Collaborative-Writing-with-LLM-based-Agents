"""
Generic JSON normalization for LLM outputs.

Different LLMs return structured data in different formats:
- Some return raw JSON: {"field": "value"}
- Some wrap in function calls: {"type": "function", "parameters": {...}}
- Some add markdown: ```json\n{...}\n```
- Some return truncated JSON

This module provides a single, model-agnostic normalization layer.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JSONNormalizer:
    """Normalize various LLM JSON output formats to standard dict."""

    @staticmethod
    def normalize(raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and normalize JSON from LLM response text.

        Handles:
        1. Clean JSON
        2. JSON wrapped in markdown code blocks
        3. JSON with text preamble/postamble
        4. Function call wrapper format (deepseek-r1)
        5. Truncated JSON (returns None to signal error)

        Returns:
            Normalized dict or None if extraction failed
        """
        if not isinstance(raw_text, str) or not raw_text:
            return None

        # Step 1: Try direct JSON parse (fastest path)
        try:
            data = json.loads(raw_text.strip())
            return JSONNormalizer._unwrap_common_formats(data)
        except json.JSONDecodeError:
            pass

        # Step 2: Extract from markdown code blocks
        markdown_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        match = re.search(markdown_pattern, raw_text)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                return JSONNormalizer._unwrap_common_formats(data)
            except json.JSONDecodeError:
                pass

        # Step 3: Find any JSON object in the text
        json_pattern = r"(\{[\s\S]*\})"
        match = re.search(json_pattern, raw_text)
        if match:
            try:
                data = json.loads(match.group(1))
                return JSONNormalizer._unwrap_common_formats(data)
            except json.JSONDecodeError:
                # Truncated JSON - log and return None
                logger.warning(
                    f"Found JSON-like structure but failed to parse. "
                    f"Likely truncated. Length: {len(match.group(1))} chars"
                )
                return None

        logger.error(f"No valid JSON found in LLM response: {raw_text[:200]}...")
        return None

    @staticmethod
    def _unwrap_common_formats(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unwrap common LLM output wrapper formats.

        Examples:
        - Function call: {"type": "function", "parameters": {...}} -> {...}
        - Tool call: {"name": "...", "arguments": {...}} -> {...}
        """
        if not isinstance(data, dict):
            return data

        # Deepseek-r1 function call format
        if data.get("type") == "function" and "parameters" in data:
            logger.debug("Unwrapping function call format (deepseek-r1)")
            return data["parameters"]

        # Generic tool call format
        if "arguments" in data and "name" in data:
            logger.debug("Unwrapping tool call format")
            return data["arguments"]

        # Return as-is if no known wrapper detected
        return data


def normalize_llm_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Convenience function for JSON normalization."""
    return JSONNormalizer.normalize(raw_text)
