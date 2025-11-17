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
            except json.JSONDecodeError as e:
                # Try to fix truncated JSON
                logger.debug(
                    f"Initial JSON parse failed: {e}. Attempting to fix truncated JSON..."
                )
                fixed_json = JSONNormalizer._attempt_fix_truncated_json(match.group(1))
                if fixed_json:
                    try:
                        data = json.loads(fixed_json)
                        logger.info("Successfully fixed truncated JSON")
                        return JSONNormalizer._unwrap_common_formats(data)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to fix truncated JSON. Original length: {len(match.group(1))} chars"
                        )
                        return None
                return None

        # Step 4: Try to find incomplete JSON at end of text
        incomplete_json_pattern = r"(\{[\s\S]*$)"
        match = re.search(incomplete_json_pattern, raw_text)
        if match:
            logger.debug("Found incomplete JSON at end of text, attempting to fix...")
            fixed_json = JSONNormalizer._attempt_fix_truncated_json(match.group(1))
            if fixed_json:
                try:
                    data = json.loads(fixed_json)
                    logger.info("Successfully fixed incomplete JSON")
                    return JSONNormalizer._unwrap_common_formats(data)
                except json.JSONDecodeError:
                    pass

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

    @staticmethod
    def _attempt_fix_truncated_json(json_str: str) -> Optional[str]:
        """
        Attempt to fix truncated JSON by adding missing closing brackets.

        Strategy:
        1. Count opening and closing brackets/braces
        2. Add missing closing characters in reverse order of opening
        3. Handle truncated strings by adding closing quote

        Returns:
            Fixed JSON string or None if unfixable
        """
        if not json_str:
            return None

        # Track what's open
        stack = []
        in_string = False
        escape_next = False

        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not in_string:
                in_string = True
            elif char == '"' and in_string:
                in_string = False
            elif not in_string:
                if char == "{":
                    stack.append("}")
                elif char == "[":
                    stack.append("]")
                elif char == "}":
                    if stack and stack[-1] == "}":
                        stack.pop()
                elif char == "]":
                    if stack and stack[-1] == "]":
                        stack.pop()

        # If we're in a string at the end, close it
        if in_string:
            json_str += '"'

        # Add all missing closing brackets/braces
        if stack:
            closing = "".join(reversed(stack))
            fixed = json_str + closing
            logger.debug(f"Added {len(stack)} closing characters: {closing}")
            return fixed

        return json_str if json_str else None


def normalize_llm_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Convenience function for JSON normalization."""
    return JSONNormalizer.normalize(raw_text)
