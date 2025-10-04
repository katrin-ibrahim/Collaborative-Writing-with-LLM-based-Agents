"""Adapter that wraps BaseEngine instances to provide LangGraph-compatible tool calling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolDescription:
    """Lightweight description holder for tools."""

    name: str
    description: str
    args_schema: Optional[Dict[str, Any]] = None


class ToolChatAdapter:
    """Wraps a BaseEngine to expose a minimal ChatModel interface with bind_tools support."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an assistant that can call tools. "
        "When responding, follow the instructions below carefully."
    )

    TOOL_INSTRUCTIONS = (
        "If you need to use a tool, respond ONLY with JSON of the form "
        '{"tool_calls":[{"name":"tool_name","arguments":{...}}]}. '
        "If you are giving the final answer, respond with text beginning with 'FINAL:'"
    )

    def __init__(self, engine, system_prompt: Optional[str] = None):
        self.engine = engine
        self.base_system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._bound_tools: List[ToolDescription] = []

    # ------------------------------------------------------------------
    # LangChain ChatModel compatibility methods
    # ------------------------------------------------------------------
    def bind_tools(self, tools: List[Any]) -> "ToolChatAdapter":
        """Return a new adapter instance bound to the provided tools."""

        adapter = ToolChatAdapter(self.engine, self.base_system_prompt)
        adapter._bound_tools = [self._describe_tool(tool) for tool in tools or []]
        return adapter

    # Alias used by some LangChain utilities
    with_structured_output = bind_tools

    def invoke(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single turn of conversation with optional tool calling."""

        prompt = self._build_prompt(messages)

        response_obj = self.engine.complete(prompt)
        response_text = self.engine.extract_content(response_obj)
        return self._parse_response(response_text)

    # ------------------------------------------------------------------
    # Prompt construction helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a prompt list suitable for BaseEngine.complete."""

        system_prompt = self._compose_system_prompt(messages)
        tool_prompt = self._tool_instruction_prompt()
        conversation = self._format_conversation(messages)

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})

        prompt_messages.append(
            {
                "role": "user",
                "content": "\n".join([tool_prompt, conversation, "Assistant:"]),
            }
        )

        return prompt_messages

    def _compose_system_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Combine provided system prompts with base instructions."""

        system_messages = [
            m.get("content", "") for m in messages if m.get("role") == "system"
        ]
        if system_messages:
            combined = "\n".join(system_messages + [self.base_system_prompt])
        else:
            combined = self.base_system_prompt
        return combined

    def _tool_instruction_prompt(self) -> str:
        if not self._bound_tools:
            return self.TOOL_INSTRUCTIONS

        tool_lines = ["Available tools:"]
        for tool in self._bound_tools:
            line = f"- {tool.name}: {tool.description or 'No description provided.'}"
            if tool.args_schema:
                line += f" (parameters: {', '.join(tool.args_schema.keys())})"
            tool_lines.append(line)

        tool_lines.append("")
        tool_lines.append(self.TOOL_INSTRUCTIONS)
        return "\n".join(tool_lines)

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        lines: List[str] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                # Already handled separately
                continue
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                if message.get("tool_calls"):
                    lines.append(
                        f"Assistant (tool call): {json.dumps(message['tool_calls'])}"
                    )
                if content:
                    lines.append(f"Assistant: {content}")
            elif role == "tool":
                tool_name = message.get("name", "tool")
                lines.append(f"Tool {tool_name} result: {content}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        response_text = response_text.strip()

        tool_payload = self._maybe_parse_tool_json(response_text)
        if tool_payload:
            tool_calls = []
            for call in tool_payload.get("tool_calls", []):
                if not isinstance(call, dict):
                    continue
                name = call.get("name")
                arguments = call.get("arguments") or call.get("args") or {}
                if name:
                    tool_calls.append({"name": name, "args": arguments})

            if tool_calls:
                return {"role": "assistant", "content": "", "tool_calls": tool_calls}

        # No tool calls -> treat as final response
        if response_text.upper().startswith("FINAL:"):
            response_text = response_text[6:].strip()

        return {"role": "assistant", "content": response_text}

    def _maybe_parse_tool_json(self, text: str) -> Optional[Dict[str, Any]]:
        candidates = [text]

        if "```" in text:
            # Extract fenced code blocks
            parts = text.split("```")
            candidates.extend(part.strip() for part in parts if part.strip())

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict) and "tool_calls" in payload:
                    return payload
            except json.JSONDecodeError:
                continue

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _describe_tool(self, tool: Any) -> ToolDescription:
        name = getattr(tool, "name", repr(tool))
        description = getattr(tool, "description", "") or ""

        args_schema = None
        schema_obj = getattr(tool, "args_schema", None)
        if schema_obj is not None:
            try:
                args_schema = schema_obj.schema().get("properties", {})
            except Exception:
                args_schema = None

        return ToolDescription(
            name=name, description=description, args_schema=args_schema
        )
