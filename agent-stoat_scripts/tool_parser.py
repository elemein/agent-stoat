"""Robust tool call extraction from LLM responses."""

import re
import json
from typing import Optional


def extract_tool_calls(response: dict) -> list[dict]:
    """
    Extract tool calls from an Ollama response.

    Tries multiple formats in order of preference:
    1. Native Ollama tool_calls field
    2. <tool_call> XML tags
    3. Markdown JSON code blocks
    4. Raw JSON objects with name/arguments

    Returns list of {"name": str, "arguments": dict}
    """
    message = response.get("message", {})

    # 1. Check for native Ollama tool_calls
    if "tool_calls" in message and message["tool_calls"]:
        return _parse_native_tool_calls(message["tool_calls"])

    # Get content for text-based parsing
    content = message.get("content", "")
    if not content:
        return []

    # Strip <think> tags before parsing
    content = _strip_think_tags(content)

    # 2. Try <tool_call> XML tags
    tool_calls = _parse_xml_tool_calls(content)
    if tool_calls:
        return tool_calls

    # 3. Try markdown JSON blocks
    tool_calls = _parse_markdown_json(content)
    if tool_calls:
        return tool_calls

    # 4. Try raw JSON objects
    tool_calls = _parse_raw_json(content)
    if tool_calls:
        return tool_calls

    return []


def _parse_native_tool_calls(tool_calls: list) -> list[dict]:
    """Parse native Ollama tool_calls format."""
    result = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            func = tc.get("function", {})
            name = func.get("name") or tc.get("name")
            args = func.get("arguments") or tc.get("arguments", {})

            # Arguments might be a JSON string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if name:
                result.append({"name": name, "arguments": args})
    return result


def _strip_think_tags(content: str) -> str:
    """Remove <think>...</think> blocks from content."""
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)


def _parse_xml_tool_calls(content: str) -> list[dict]:
    """Extract tool calls from <tool_call>...</tool_call> tags."""
    result = []

    # Match <tool_call>...</tool_call> blocks
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        parsed = _try_parse_tool_json(match)
        if parsed:
            result.append(parsed)

    return result


def _parse_markdown_json(content: str) -> list[dict]:
    """Extract tool calls from ```json ... ``` blocks."""
    result = []

    # Match ```json ... ``` or ``` ... ``` blocks
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        parsed = _try_parse_tool_json(match)
        if parsed:
            result.append(parsed)

    return result


def _parse_raw_json(content: str) -> list[dict]:
    """Try to find raw JSON objects that look like tool calls."""
    result = []

    # Look for JSON objects with "name" field
    # This regex finds balanced braces (simple approach)
    pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*[^{}]*\}'
    matches = re.findall(pattern, content)

    for match in matches:
        parsed = _try_parse_tool_json(match)
        if parsed:
            result.append(parsed)

    # Also try to find more complex nested JSON
    if not result:
        # Find potential JSON starting points
        for i, char in enumerate(content):
            if char == '{':
                json_str = _extract_json_object(content, i)
                if json_str:
                    parsed = _try_parse_tool_json(json_str)
                    if parsed:
                        result.append(parsed)

    return result


def _extract_json_object(content: str, start: int) -> Optional[str]:
    """Extract a JSON object starting at the given position."""
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, min(start + 5000, len(content))):  # Limit search
        char = content[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return content[start:i + 1]

    return None


def _try_parse_tool_json(text: str) -> Optional[dict]:
    """Try to parse text as a tool call JSON."""
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common issues
        text = _fix_json_string(text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    # Extract name and arguments
    name = data.get("name")
    if not name:
        return None

    # Arguments can be in "arguments", "parameters", or "args"
    arguments = (
        data.get("arguments") or
        data.get("parameters") or
        data.get("args") or
        {}
    )

    # Arguments might be a JSON string
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}

    return {"name": name, "arguments": arguments}


def _fix_json_string(text: str) -> str:
    """Attempt to fix common JSON formatting issues."""
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # Fix unquoted keys (simple cases)
    text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

    return text


def get_response_content(response: dict) -> str:
    """Get the text content from a response, stripping think tags."""
    content = response.get("message", {}).get("content", "")
    return _strip_think_tags(content).strip()
