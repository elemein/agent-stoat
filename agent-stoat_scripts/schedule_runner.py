"""schedule_runner.py — Lightweight schedule poller for Agent Stoat.

Reads .agent-stoat/SCHEDULE.md every POLL_INTERVAL seconds.
When a task is due, fires a minimal LLM tick with that task description.
Updates SCHEDULE.md to track next-run times and mark one-time tasks done.

Schedule format (one entry per line):
    YYYY-MM-DD HH:MM | description          one-time
    daily HH:MM      | description          repeats every day
    every Xh         | description          repeats every X hours
    every Xm         | description          repeats every X minutes

Lines starting with # are ignored.
The runner appends [next: YYYY-MM-DD HH:MM] to recurring lines for tracking.
One-time tasks are marked [done: YYYY-MM-DD HH:MM] after firing.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

DIM = "\033[2m"
RESET = "\033[0m"


def _log(msg: str) -> None:
    """Print a dim timestamped scheduler log line to stdout."""
    ts = datetime.now().strftime("%H:%M:%S")
    sys.stdout.write(f"\r\033[2K{DIM}  [Schedule {ts}] {msg}{RESET}\n")
    sys.stdout.flush()

POLL_INTERVAL = 2 * 60  # seconds between polls

_schedule_thread: threading.Thread | None = None
_stop_event = threading.Event()
_running = False

# Optional function(channel_id: int, text: str) for direct Discord delivery.
# Set by agent-stoat.py at startup so this module stays decoupled from discord_bridge.
_discord_send_fn: Callable[[int, str], None] | None = None

# Tools allowed during a scheduled tick (same as heartbeat)
_SCHEDULE_TOOL_NAMES = {"read_memory", "update_memory", "read_schedule", "update_schedule", "get_current_time", "read_daily_log", "append_daily_log", "clear_context"}

_DT_FMT = "%Y-%m-%d %H:%M"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def set_discord_send_fn(fn: Callable[[int, str], None]) -> None:
    """Inject the Discord channel-send function (discord_bridge.send_to_channel)."""
    global _discord_send_fn
    _discord_send_fn = fn


def get_schedule_path(data_dir: str) -> str:
    return os.path.join(data_dir, "SCHEDULE.md")


def ensure_default_schedule(data_dir: str) -> None:
    path = get_schedule_path(data_dir)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "# Stoat Schedule\n"
                "#\n"
                "# Add scheduled tasks below. Stoat checks this file every few minutes.\n"
                "#\n"
                "# Formats:\n"
                "#   YYYY-MM-DD HH:MM  | task description     (one-time)\n"
                "#   daily HH:MM       | task description     (repeats daily)\n"
                "#   every Xh          | task description     (repeats every X hours)\n"
                "#   every Xm          | task description     (repeats every X minutes)\n"
                "#\n"
                "# Lines starting with # are ignored.\n"
                "# Do not hand-edit [next: ...] or [done: ...] annotations.\n"
            )


# ---------------------------------------------------------------------------
# Line parser
# ---------------------------------------------------------------------------

def _parse_line(line: str) -> dict | None:
    """Parse one schedule line. Returns a dict or None if not actionable."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    # Skip already-done one-time tasks
    if re.search(r"\[done(?::[^\]]+)?\]", stripped):
        return None

    # Extract [next: YYYY-MM-DD HH:MM] annotation
    next_match = re.search(r"\[next:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]", stripped)
    next_dt: datetime | None = None
    if next_match:
        try:
            next_dt = datetime.strptime(next_match.group(1), _DT_FMT)
        except ValueError:
            pass

    # Remove annotations to get the clean schedule spec
    clean = re.sub(r"\[[^\]]+\]", "", stripped).strip()

    parts = clean.split(" | ", 1)
    if len(parts) != 2:
        return None
    schedule_spec = parts[0].strip()
    description = parts[1].strip()
    if not description:
        return None

    # Extract [ch:CHANNEL_ID] annotation if present — used for Discord routing.
    channel_id: int | None = None
    ch_match = re.search(r"\[ch:(\d+)\]", description)
    if ch_match:
        try:
            channel_id = int(ch_match.group(1))
        except ValueError:
            pass
        description = re.sub(r"\s*\[ch:\d+\]", "", description).strip()

    # Tasks prefixed with "message:" are delivered directly — no LLM needed.
    direct = False
    if description.lower().startswith("message:"):
        description = description[len("message:"):].strip()
        direct = True

    now = datetime.now()

    # --- One-time: YYYY-MM-DD HH:MM ---
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", schedule_spec):
        try:
            fire_dt = datetime.strptime(schedule_spec, _DT_FMT)
        except ValueError:
            return None
        return {
            "kind": "once",
            "description": description,
            "next_run": fire_dt,
            "direct": direct,
            "channel_id": channel_id,
        }

    # --- Daily: daily HH:MM ---
    m = re.match(r"^daily\s+(\d{1,2}):(\d{2})$", schedule_spec, re.IGNORECASE)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        if next_dt:
            fire_dt = next_dt
        else:
            fire_dt = now.replace(hour=h, minute=mn, second=0, microsecond=0)
            # Only push to tomorrow if missed by more than one poll interval.
            # Tasks added close to their fire time should still be caught on the
            # next poll even if the exact minute has just passed.
            if fire_dt < now - timedelta(seconds=POLL_INTERVAL):
                fire_dt += timedelta(days=1)
        return {
            "kind": "daily",
            "description": description,
            "next_run": fire_dt,
            "hour": h,
            "minute": mn,
            "direct": direct,
            "channel_id": channel_id,
        }

    # --- Interval: every Xh or every Xm ---
    m = re.match(r"^every\s+(\d+)(h|m)$", schedule_spec, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        interval = timedelta(hours=n) if unit == "h" else timedelta(minutes=n)
        if next_dt:
            fire_dt = next_dt
        else:
            fire_dt = now + interval  # first tick after one interval
        return {
            "kind": "every",
            "description": description,
            "next_run": fire_dt,
            "interval": interval,
            "direct": direct,
            "channel_id": channel_id,
        }

    return None


# ---------------------------------------------------------------------------
# File updater
# ---------------------------------------------------------------------------

def _update_line_next(line: str, next_dt: datetime) -> str:
    """Replace or append a [next: ...] annotation on a line."""
    tag = f"[next: {next_dt.strftime(_DT_FMT)}]"
    if re.search(r"\[next:[^\]]+\]", line):
        return re.sub(r"\[next:[^\]]+\]", tag, line.rstrip()) + "\n"
    return line.rstrip() + f" {tag}\n"


def _mark_line_done(line: str, now: datetime) -> str:
    """Replace [next: ...] or append [done: ...] to a one-time task line."""
    tag = f"[done: {now.strftime(_DT_FMT)}]"
    cleaned = re.sub(r"\[[^\]]+\]", "", line).rstrip()
    return cleaned + f" {tag}\n"


# ---------------------------------------------------------------------------
# LLM tick
# ---------------------------------------------------------------------------

def _run_scheduled_task(
    description: str,
    now: datetime,
    data_dir: str,
    llm_caller: Callable,
    tool_executor: Callable,
    alert_callback: Callable[[str], None],
    tools: list[dict],
) -> None:
    """Fire one LLM tick for a due scheduled task."""
    # Import here to avoid circular imports
    from persistent_mode import (  # type: ignore
        _read_file_safe,
        get_memory_path,
        HEARTBEAT_OK,
        _load_heartbeat_prompt,
    )

    memory = _read_file_safe(get_memory_path(data_dir))
    time_str = now.strftime("%A, %Y-%m-%d %H:%M")

    parts = [
        f"Current time: {time_str}",
        "",
        "## Scheduled Task",
        description,
    ]
    if memory:
        parts += ["", "## Your Memory (MEMORY.md)", memory]
    parts += [
        "",
        f"Complete the task above. If there is nothing to report, respond only with: {HEARTBEAT_OK}",
        "If you have something to surface to the user, respond with a brief message and omit HEARTBEAT_OK.",
        "Use update_memory to record anything worth remembering.",
    ]

    schedule_tools = [
        t for t in tools
        if t.get("function", {}).get("name") in _SCHEDULE_TOOL_NAMES
    ]

    messages = [
        {"role": "system", "content": _load_heartbeat_prompt()},
        {"role": "user", "content": "\n".join(parts)},
    ]

    try:
        response = llm_caller(
            messages,
            tools=schedule_tools or None,
            print_output=False,
            output_callback=None,
        )
    except Exception as e:
        alert_callback(f"[Schedule] LLM error: {e}")
        return

    content = (response.get("message") or {}).get("content") or ""
    tool_calls = (response.get("message") or {}).get("tool_calls") or []

    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        args = func.get("arguments", {})
        if name and tool_executor:
            try:
                tool_executor(name, args)
            except Exception as e:
                alert_callback(f"[Schedule] Tool error ({name}): {e}")

    stripped = content.strip()
    is_ok = (
        stripped == HEARTBEAT_OK
        or stripped.startswith(HEARTBEAT_OK + "\n")
        or stripped.endswith("\n" + HEARTBEAT_OK)
    )
    if not is_ok and stripped:
        label = description[:40] + ("..." if len(description) > 40 else "")
        alert_callback(f"[Schedule: {label}]\n{stripped}")


# ---------------------------------------------------------------------------
# Poll tick
# ---------------------------------------------------------------------------

def _poll_tick(
    data_dir: str,
    llm_caller: Callable,
    tool_executor: Callable,
    alert_callback: Callable[[str], None],
    tools: list[dict],
    turn_lock: threading.Lock,
) -> None:
    """Read SCHEDULE.md, fire any due tasks, update the file."""
    path = get_schedule_path(data_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    except Exception as e:
        alert_callback(f"[Schedule] Could not read SCHEDULE.md: {e}")
        return

    now = datetime.now()
    modified = False

    # Count actionable entries for the log
    entries = [(i, line, _parse_line(line)) for i, line in enumerate(lines)]
    active = [e for e in entries if e[2] is not None]
    due = [e for e in active if e[2]["next_run"] <= now]
    _log(f"poll — {len(active)} task(s), {len(due)} due")

    for i, line, entry in entries:
        if entry is None:
            continue
        if entry["next_run"] > now:
            continue

        # Task is due — fire it
        _log(f"firing: {entry['description'][:60]}")
        task_ran = False

        if entry.get("direct"):
            # Direct message — deliver immediately, no LLM needed.
            # Route to the embedded channel ID if present, else fall back to alert_callback.
            cid = entry.get("channel_id")
            if cid and _discord_send_fn is not None:
                _discord_send_fn(cid, entry["description"])
            else:
                alert_callback(entry["description"])
            task_ran = True
        else:
            # LLM task — acquire turn lock and run a model tick
            acquired = turn_lock.acquire(timeout=10)
            if acquired:
                try:
                    _run_scheduled_task(
                        entry["description"], now,
                        data_dir, llm_caller, tool_executor, alert_callback, tools,
                    )
                    task_ran = True
                finally:
                    turn_lock.release()
            else:
                alert_callback(
                    f"[Schedule] Skipped '{entry['description'][:40]}' — turn lock busy, will retry next poll"
                )

        if not task_ran:
            continue

        # Update the line now that the task has actually run
        if entry["kind"] == "once":
            lines[i] = _mark_line_done(line, now)
        elif entry["kind"] == "daily":
            h, mn = entry["hour"], entry["minute"]
            next_dt = now.replace(hour=h, minute=mn, second=0, microsecond=0)
            if next_dt <= now:
                next_dt += timedelta(days=1)
            lines[i] = _update_line_next(line, next_dt)
        else:  # every
            interval = entry["interval"]
            # Jump forward by however many intervals have elapsed, then one more
            elapsed = (now - entry["next_run"]).total_seconds()
            steps = max(0, int(elapsed / interval.total_seconds())) + 1
            next_dt = entry["next_run"] + interval * steps
            lines[i] = _update_line_next(line, next_dt)

        modified = True

    if modified:
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(path), suffix=".tmp", text=True
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.writelines(lines)
            os.replace(tmp_path, path)
        except Exception as e:
            alert_callback(f"[Schedule] Could not write SCHEDULE.md: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


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
    poll_interval: int = POLL_INTERVAL,
) -> None:
    """Start the schedule polling thread."""
    global _schedule_thread, _running

    if _running:
        return

    ensure_default_schedule(data_dir)
    _stop_event.clear()
    _running = True

    def _loop():
        while not _stop_event.is_set():
            _poll_tick(data_dir, llm_caller, tool_executor, alert_callback, tools, turn_lock)
            _stop_event.wait(timeout=poll_interval)

    _schedule_thread = threading.Thread(target=_loop, daemon=True, name="stoat-schedule")
    _schedule_thread.start()


def stop() -> None:
    global _running
    _running = False
    _stop_event.set()
    if _schedule_thread:
        _schedule_thread.join(timeout=3)


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
    """Run a poll tick immediately from the calling thread."""
    _poll_tick(data_dir, llm_caller, tool_executor, alert_callback, tools, turn_lock)
