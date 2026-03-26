"""ChatEngine — single-agent chat loop with tool calling.

This is the core execution engine. It takes a user message, sends it to the
LLM with the system prompt and conversation history, executes any tool calls
the model makes, and loops until the model responds without a tool call (done)
or the iteration limit is reached.

Optional context compaction summarizes old tool interactions when the context
window fills past a configurable threshold (default 80%).
"""

import json
import os
import re
import sys
import threading
import time
from typing import Callable

from tool_parser import extract_tool_calls, get_response_content

# ANSI colors
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def _print(text: str, color: str = RESET) -> None:
    print(f"{color}{text}{RESET}")


def _truncate(text: str, max_len: int = 300) -> str:
    """Collapse whitespace and truncate for display."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len] + "..." if len(text) > max_len else text


class _Spinner:
    """Animated in-place spinner with elapsed time for long-running operations."""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "Working"):
        self.label = label
        self._active = False
        self._thread = None
        self._start = 0.0

    def __enter__(self):
        self._active = True
        self._start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._active = False
        if self._thread:
            self._thread.join(timeout=1)
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    def _run(self):
        i = 0
        while self._active:
            elapsed = time.time() - self._start
            frame = self._FRAMES[i % len(self._FRAMES)]
            sys.stdout.write(f"\r{DIM}  {frame} {self.label}... {elapsed:.0f}s{RESET}")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1


class ChatEngine:
    """Single-agent chat loop with tool calling and optional context compaction.

    Args:
        llm_caller: Function matching chat_stream(messages, tools=, print_output=, color=).
        tool_executor: Function matching execute_tool(name, args) -> str.
        is_dangerous: Returns True if a tool name requires permission.
        get_permission: Returns True (allow), False (deny), or None (ask user).
        conv_history: Mutable list shared with the REPL for persistence across turns.
        get_usage_pct: Optional callback returning context usage as a percentage (0-100).
    """

    def __init__(
        self,
        llm_caller: Callable,
        tool_executor: Callable,
        is_dangerous: Callable,
        get_permission: Callable,
        conv_history: list,
        get_usage_pct: Callable = None,
        ask_permission: Callable = None,
    ):
        self.llm_caller = llm_caller
        self.tool_executor = tool_executor
        self.is_dangerous = is_dangerous
        self.get_permission = get_permission
        self.conv_history = conv_history
        self.get_usage_pct = get_usage_pct
        # Optional override for the interactive "allow this tool?" prompt.
        # Signature: ask_permission(tool_name: str, args: dict) -> bool
        # Defaults to a plain input() prompt if None.
        self.ask_permission = ask_permission

    def run(
        self,
        goal: str,
        prompt: str,
        tools: list[dict],
        max_iterations: int = 50,
        compaction: bool = False,
        compact_threshold: int = 80,
        keep_recent: int = 16,
        output_callback: Callable = None,
    ) -> None:
        """Run the chat loop for one user message.

        The model calls tools in a loop until it produces a plain text response
        (no tool calls) or hits max_iterations. History is synced back to
        conv_history when the loop exits.

        Args:
            compact_threshold: Context usage percentage (0-100) at which compaction
                triggers. Only used if compaction=True and get_usage_pct is set.
        """
        keep_recent = max(1, keep_recent)  # guard against zero/negative
        # Inject working directory listing into system prompt
        messages = [{"role": "system", "content": prompt + self._dir_info()}]

        # Restore prior conversation or start fresh
        if self.conv_history:
            for msg in self.conv_history:
                if msg.get("role") != "system":
                    messages.append(msg)
        else:
            messages.append({"role": "user", "content": goal})

        # Main tool-calling loop
        for _ in range(max_iterations):
            if compaction:
                messages = self._compact_messages(messages, keep_recent, compact_threshold)

            response = self.llm_caller(
                messages, tools=tools, print_output=True, color=GREEN,
                output_callback=output_callback,
            )

            if response.get("interrupted"):
                self._save_history(messages)
                return

            tool_calls = extract_tool_calls(response)
            content = get_response_content(response)

            # No tool calls → model is done
            if not tool_calls:
                messages.append({"role": "assistant", "content": content or ""})
                self._save_history(messages)
                return

            # Record assistant message with its tool calls
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in tool_calls
                ],
            })

            # Execute each tool call
            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]

                _print(f"\n  >> {name}", YELLOW)
                for key, value in args.items():
                    display = str(value)[:80] + "..." if len(str(value)) > 80 else str(value)
                    _print(f"     {key}: {display}", YELLOW)

                # Permission gate — check all tools, prompt only for dangerous ones
                perm = self.get_permission(name)
                if perm is False:
                    _print("  -- (Blocked)", RED)
                    messages.append({"role": "tool", "content": "(Blocked by policy)"})
                    continue
                elif perm is None and self.is_dangerous(name):
                    if self.ask_permission is not None:
                        allowed = self.ask_permission(name, args)
                    else:
                        try:
                            answer = input(f"  Allow {name}? [y/n]: ").strip().lower()
                        except EOFError:
                            answer = 'n'
                        allowed = (answer == 'y')
                    if not allowed:
                        _print("  -- (Denied)", RED)
                        messages.append({"role": "tool", "content": "(Denied by user)"})
                        continue

                with _Spinner(name):
                    result = self.tool_executor(name, args)
                _print(f"  -- {_truncate(result)}", GREEN)
                messages.append({"role": "tool", "content": result})

        # Hit max iterations
        self._save_history(messages)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _dir_info() -> str:
        """Build a working directory summary to append to the system prompt."""
        cwd = os.getcwd()
        try:
            entries = sorted(os.listdir(cwd))[:30]
            if entries:
                return f"\n\nCurrent working directory: `{cwd}`\nFiles here: {', '.join(entries)}"
            return f"\n\nCurrent working directory: `{cwd}` (empty)"
        except Exception:
            return f"\n\nCurrent working directory: `{cwd}`"

    def _save_history(self, messages: list[dict]) -> None:
        """Sync messages back to the shared conv_history (excluding system prompt)."""
        self.conv_history.clear()
        self.conv_history.extend(m for m in messages if m.get("role") != "system")

    def _compact_messages(
        self,
        messages: list[dict],
        keep_recent: int = 16,
        threshold: int = 80,
    ) -> list[dict]:
        """Compress old assistant+tool pairs into one-line summaries.

        Triggers only when context usage exceeds the threshold percentage.
        Splits messages into [system] + [old] + [recent]. Keeps system and
        recent verbatim, replaces old assistant+tool pairs with short summaries.
        """
        # Only compact when context is filling up
        if self.get_usage_pct:
            usage_pct = self.get_usage_pct()
            if usage_pct < threshold:
                return messages

        # Need at least some old messages to compact
        system = messages[0] if messages and messages[0].get("role") == "system" else None
        start = 1 if system else 0
        cutoff = max(start, len(messages) - keep_recent)
        old = messages[start:cutoff]
        recent = messages[cutoff:]

        compactable = sum(1 for m in old if m.get("role") in ("assistant", "tool"))
        if compactable == 0:
            return messages

        _print(f"\n  [Compaction] Context at {min(self.get_usage_pct(), 100):.0f}% — "
               f"summarizing {compactable} older messages "
               f"(keeping last {keep_recent} verbatim)", DIM)

        compacted = []
        if system:
            compacted.append(system)

        summaries_log = []
        i = 0
        while i < len(old):
            msg = old[i]
            role = msg.get("role", "")

            if role == "user":
                # Always keep user messages intact
                compacted.append(msg)
                i += 1

            elif role == "assistant" and msg.get("tool_calls"):
                # Summarize this assistant + following tool result(s)
                summary = self._summarize_tool_turn(msg)
                compacted.append({"role": "assistant", "content": summary})
                summaries_log.append(summary)

                # Skip the following tool result messages
                i += 1
                while i < len(old) and old[i].get("role") == "tool":
                    tool_content = old[i].get("content", "")
                    if "error" in tool_content.lower():
                        err = f"(Tool error: {tool_content[:120]})"
                        compacted.append({"role": "assistant", "content": err})
                        summaries_log.append(err)
                    i += 1

            elif role == "assistant":
                # Plain text response — truncate if long
                text = msg.get("content", "")
                if len(text) > 200:
                    text = text[:200] + "..."
                compacted.append({"role": "assistant", "content": text})
                i += 1

            else:
                i += 1  # Skip orphaned tool messages

        compacted.extend(recent)

        for s in summaries_log:
            _print(f"    > {s[:120]}", DIM)
        _print(f"  [Compaction] {len(messages)} msgs -> {len(compacted)} msgs", DIM)

        return compacted

    @staticmethod
    def _summarize_tool_turn(msg: dict) -> str:
        """Create a one-line summary of an assistant message that made tool calls."""
        summaries = []
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            name = func.get("name", "?")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if name == "write_file":
                content = args.get("content", "")
                summaries.append(f"Wrote {args.get('path', '?')} ({len(content)} chars)")
            elif name == "edit_file":
                summaries.append(f"Edited {args.get('path', '?')}")
            elif name == "read_file":
                summaries.append(f"Read {args.get('path', '?')}")
            elif name == "shell":
                summaries.append(f"Ran: {args.get('command', '?')[:60]}")
            elif name == "list_dir":
                summaries.append("Listed directory")
            else:
                summaries.append(f"Called {name}")

        # Prepend brief assistant reasoning if present
        text = (msg.get("content") or "")[:100].replace("\n", " ").strip()
        if text:
            return f"{text} -> {'; '.join(summaries)}"
        return "; ".join(summaries)
