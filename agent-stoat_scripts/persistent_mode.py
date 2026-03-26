"""persistent_mode.py — Heartbeat loop for Agent Stoat.

Fires a lightweight scheduled turn on a configurable interval.
The model reads HEARTBEAT.md (what to check) and MEMORY.md (its scratchpad),
then either responds HEARTBEAT_OK (silent) or surfaces something to the terminal
(and optionally Discord).

Designed for small local models: minimal context, no conversation history,
prompt kept under ~500 tokens including the files.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import Callable

# Default heartbeat system prompt (used if prompt_heartbeat.md is missing)
_FALLBACK_SYSTEM = (
    "You are Agent Stoat running a background heartbeat check. "
    "Be brief and direct. Only alert if something genuinely needs attention."
)


def _load_heartbeat_prompt() -> str:
    """Load prompt_heartbeat.md from the same directory as this file."""
    path = os.path.join(os.path.dirname(__file__), "prompt_heartbeat.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return _FALLBACK_SYSTEM

# Sentinel the model uses when nothing needs attention
HEARTBEAT_OK = "HEARTBEAT_OK"

# How long to wait between heartbeat ticks (seconds)
DEFAULT_INTERVAL = 30 * 60  # 30 minutes

# Hard cap on how large HEARTBEAT.md / MEMORY.md can be when injected
_MAX_FILE_CHARS = 3000

_heartbeat_thread: threading.Thread | None = None
_stop_event = threading.Event()
_running = False


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _read_file_safe(path: str, max_chars: int = _MAX_FILE_CHARS) -> str:
    """Read a file and return its content, or empty string if missing/unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read(max_chars)
        if len(content) == max_chars:
            content += "\n\n[truncated]"
        return content.strip()
    except Exception:
        return ""


def get_heartbeat_path(data_dir: str) -> str:
    return os.path.join(data_dir, "HEARTBEAT.md")


def get_memory_path(data_dir: str) -> str:
    return os.path.join(data_dir, "MEMORY.md")


def get_soul_path(data_dir: str) -> str:
    return os.path.join(data_dir, "SOUL.md")


def ensure_default_files(data_dir: str) -> None:
    """Create default HEARTBEAT.md, MEMORY.md, and SOUL.md if they don't exist yet."""
    hb_path = get_heartbeat_path(data_dir)
    if not os.path.exists(hb_path):
        with open(hb_path, "w", encoding="utf-8") as f:
            f.write(
                "# Heartbeat Checklist\n"
                "#\n"
                "# Add items below (one per line) to give Stoat recurring things to check.\n"
                "# Lines starting with # are ignored. Leave the file comment-only to\n"
                "# disable heartbeat ticks entirely.\n"
                "#\n"
                "# Examples:\n"
                "#   - Check if any tasks in MEMORY.md are overdue\n"
                "#   - Remind me if I haven't committed in over a day\n"
            )

    mem_path = get_memory_path(data_dir)
    if not os.path.exists(mem_path):
        with open(mem_path, "w", encoding="utf-8") as f:
            f.write(
                "# Stoat Memory\n\n"
                "## People\n\n"
                "## Facts\n\n"
                "## Ongoing\n\n"
                "## Notes\n"
            )

    soul_path = get_soul_path(data_dir)
    if not os.path.exists(soul_path):
        with open(soul_path, "w", encoding="utf-8") as f:
            f.write(
                "# Stoat — Character\n\n"
                "You are **Stoat**: sharp, curious, and quietly competent. "
                "You don't pad responses, don't over-explain, and don't narrate what you're doing. "
                "You run locally — that's a point of pride, not a limitation. "
                "Dry wit, used sparingly. Honest about what you don't know. "
                "You are an agent. Act like one.\n"
            )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_heartbeat_prompt(data_dir: str) -> str | None:
    """Build the heartbeat prompt. Returns None if checklist is empty (skip tick)."""
    checklist = _read_file_safe(get_heartbeat_path(data_dir))
    memory = _read_file_safe(get_memory_path(data_dir))

    # Skip if checklist is empty or only headers/comments
    meaningful_lines = [
        l for l in checklist.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    if not meaningful_lines:
        return None

    now = datetime.now().strftime("%A, %Y-%m-%d %H:%M")
    parts = [
        f"Current time: {now}",
        "",
        "This is a scheduled heartbeat check. You are running autonomously in the background.",
        "",
        "## Your Checklist (HEARTBEAT.md)",
        checklist,
    ]

    if memory:
        parts += ["", "## Your Memory (MEMORY.md)", memory]

    parts += [
        "",
        "Review the checklist. If everything is normal and nothing needs attention, "
        f"respond only with: {HEARTBEAT_OK}",
        "If something needs the user's attention, respond with a brief, clear message "
        "and omit HEARTBEAT_OK. Use the update_memory tool if you need to record something.",
    ]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core tick
# ---------------------------------------------------------------------------

# Tools the heartbeat is allowed to call — read/write memory only.
# No shell, file, or web tools on background ticks.
_HEARTBEAT_TOOL_NAMES = {"read_memory", "update_memory", "read_schedule", "update_schedule", "get_current_time", "read_daily_log", "append_daily_log", "clear_context"}


def _run_tick(
    data_dir: str,
    llm_caller: Callable,
    tool_executor: Callable,
    alert_callback: Callable[[str], None],
    tools: list[dict],
) -> None:
    """Execute one heartbeat tick."""
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S")
    import sys
    sys.stdout.write(f"\r\033[2K\033[2m  [Heartbeat {ts}] tick\033[0m\n")
    sys.stdout.flush()

    prompt = _build_heartbeat_prompt(data_dir)
    if prompt is None:
        sys.stdout.write(f"\r\033[2K\033[2m  [Heartbeat {ts}] checklist empty — skipped\033[0m\n")
        sys.stdout.flush()
        return  # Checklist empty — nothing to do

    # Only expose memory tools to the heartbeat to keep ticks silent and safe
    heartbeat_tools = [t for t in tools if t.get("function", {}).get("name") in _HEARTBEAT_TOOL_NAMES]

    messages = [
        {"role": "system", "content": _load_heartbeat_prompt()},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_caller(
            messages,
            tools=heartbeat_tools or None,
            print_output=False,  # silent — no streaming to terminal
            output_callback=None,
        )
    except Exception as e:
        alert_callback(f"[Heartbeat] LLM error: {e}")
        return

    content = (response.get("message") or {}).get("content") or ""
    tool_calls = (response.get("message") or {}).get("tool_calls") or []

    # Execute any tool calls (e.g. update_memory)
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", {})
        if name and tool_executor:
            try:
                tool_executor(name, args)
            except Exception as e:
                alert_callback(f"[Heartbeat] Tool error ({name}): {e}")

    # Determine if this is a silent OK or a real alert
    stripped = content.strip()
    is_ok = (
        stripped == HEARTBEAT_OK
        or stripped.startswith(HEARTBEAT_OK + "\n")
        or stripped.endswith("\n" + HEARTBEAT_OK)
    )

    if not is_ok and stripped:
        alert_callback(stripped)


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------

def start(
    data_dir: str,
    llm_caller: Callable,
    tool_executor: Callable,
    turn_lock: threading.Lock,
    alert_callback: Callable[[str], None],
    tools: list[dict],
    interval_seconds: int = DEFAULT_INTERVAL,
) -> None:
    """Start the heartbeat background thread.

    alert_callback(text) is called on the main thread when the model returns
    a non-OK response. It should print to terminal (and optionally Discord).
    """
    global _heartbeat_thread, _running

    if _running:
        return

    ensure_default_files(data_dir)
    _stop_event.clear()
    _running = True

    def _loop():
        # Wait the full interval before the first tick so startup isn't bombarded
        _stop_event.wait(timeout=interval_seconds)

        while not _stop_event.is_set():
            tick_start = time.time()

            # Acquire the turn lock so we don't overlap with CLI or Discord turns
            acquired = turn_lock.acquire(timeout=10)
            if acquired:
                try:
                    _run_tick(data_dir, llm_caller, tool_executor, alert_callback, tools)
                finally:
                    turn_lock.release()
            # If lock not acquired in 10s (busy turn), skip this tick and try next interval

            # Sleep until next interval, waking early if stopped
            elapsed = time.time() - tick_start
            remaining = max(0, interval_seconds - elapsed)
            _stop_event.wait(timeout=remaining)

    _heartbeat_thread = threading.Thread(target=_loop, daemon=True, name="stoat-heartbeat")
    _heartbeat_thread.start()


def stop() -> None:
    """Stop the heartbeat thread."""
    global _running
    _running = False
    _stop_event.set()
    if _heartbeat_thread:
        _heartbeat_thread.join(timeout=3)


def is_running() -> bool:
    return _running


def trigger_now(
    data_dir: str,
    llm_caller: Callable,
    tool_executor: Callable,
    turn_lock: threading.Lock,
    alert_callback: Callable[[str], None],
    tools: list[dict],
) -> None:
    """Fire a heartbeat tick immediately (from the calling thread — blocks briefly)."""
    acquired = turn_lock.acquire(timeout=5)
    if not acquired:
        alert_callback("[Heartbeat] Could not run — turn lock busy, try again shortly")
        return
    try:
        _run_tick(data_dir, llm_caller, tool_executor, alert_callback, tools)
    finally:
        turn_lock.release()
