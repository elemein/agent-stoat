"""Tool definitions and execution for Agent Stoat.

Provides the tool schema (TOOLS) sent to the LLM, the matching implementation
functions, and a permission system for gating dangerous operations (write_file,
edit_file, shell). The execute_tool() dispatcher is the single entry point used
by ChatEngine.
"""

import fnmatch
import os
import re
import subprocess
import sys
import threading
import urllib.request
import urllib.error
import urllib.parse
import html
from html.parser import HTMLParser

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_DIR = os.path.dirname(_SCRIPTS_DIR)
AGENT_DATA_DIR = os.path.join(_AGENT_DIR, ".agent-stoat")
WORKING_DIR = os.path.join(_AGENT_DIR, "agent-stoat_working-dir")
os.makedirs(AGENT_DATA_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# Tool definitions in OpenAI function-calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents. Large files are truncated — use start_line/end_line to read specific sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "start_line": {"type": "integer", "description": "First line to read (1-based, inclusive)"},
                    "end_line": {"type": "integer", "description": "Last line to read (1-based, inclusive)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a unique exact string in a file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old": {"type": "string", "description": "Exact string to find (must be unique)"},
                    "new": {"type": "string", "description": "Replacement string"}
                },
                "required": ["path", "old", "new"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command and return output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return clean text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and folders in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current)"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files by glob pattern recursively (e.g. '*.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '*.py')"},
                    "path": {"type": "string", "description": "Directory to search (default: current)"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text/regex inside files recursively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex to search for"},
                    "path": {"type": "string", "description": "Directory to search (default: current)"},
                    "glob": {"type": "string", "description": "File filter (e.g. '*.py')"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_schedule",
            "description": "Read Stoat's schedule file (SCHEDULE.md). Use this to see what tasks are currently scheduled.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_schedule",
            "description": (
                "Overwrite Stoat's schedule file (SCHEDULE.md) with new content. "
                "Valid formats per line:\n"
                "  YYYY-MM-DD HH:MM | task          (one-time)\n"
                "  daily HH:MM | task               (repeats daily)\n"
                "  every Xh | task                  (repeats every X hours)\n"
                "  every Xm | task                  (repeats every X minutes)\n"
                "To deliver a message directly to the user, prefix the task with 'message:'\n"
                "Always include [ch:CHANNEL_ID] to route to the right Discord channel, and @mention the requesting user in the message.\n"
                "  Example: 2026-03-26 14:00 | message: <@123456789> Hey! Your build finished. [ch:987654321]\n"
                "The channel_id and user mention are in the [Discord channel_id:NNNN ... mention:<@ID>] prefix of incoming messages.\n"
                "Without 'message:', the task runs as an LLM reasoning tick.\n"
                "Do not include or modify [next: ...] or [done: ...] annotations — those are managed automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Full new content for SCHEDULE.md"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current local date and time. Use this whenever you need to know the current time — no shell or permissions required.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_memory",
            "description": "Read Stoat's persistent memory file (MEMORY.md). Use this to recall notes, tasks, or context saved across sessions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Overwrite Stoat's persistent memory file (MEMORY.md) with new content. Keep it concise — under 100 lines. Use markdown headers to organise sections (## People, ## Facts, ## Ongoing, ## Notes).",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Full new content for MEMORY.md"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_daily_log",
            "description": "Read the current daily log — a running record of notable events since the last 4 AM reset.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_daily_log",
            "description": "Append a timestamped entry to the daily log. Use during conversation to note anything worth remembering — facts learned, tasks completed, decisions made.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry": {"type": "string", "description": "The note to log (one line, plain text)"}
                },
                "required": ["entry"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_context",
            "description": "Flag that conversation context should be cleared on the next user interaction. Used by the 4 AM daily reset to start fresh each day.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
]


def read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    """Read and return the contents of a file, optionally limited to a line range."""
    try:
        path = os.path.expanduser(path)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)

        if start_line is not None or end_line is not None:
            lo = max(1, start_line or 1) - 1
            hi = min(total_lines, end_line or total_lines)
            if lo >= total_lines:
                return f"Error: start_line ({lo + 1}) exceeds file length ({total_lines} lines)."
            lines = lines[lo:hi]
            header = f"[Lines {lo + 1}–{lo + len(lines)} of {total_lines}]\n"
        else:
            lo = 0
            header = ""

        content = "".join(lines)

        if len(content) > 5000:
            shown_lines = lo + content[:5000].count("\n")
            content = content[:5000] + (
                f"\n\n[Truncated — showing lines {lo + 1}–{shown_lines} of {total_lines}. "
                f"Call read_file with start_line={shown_lines + 1} to continue.]"
            )
            return content

        return header + content

    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        path = os.path.expanduser(path)
        # Create parent directories if they don't exist
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _find_normalized(content: str, old: str):
    """
    Try to find old in content ignoring per-line leading/trailing whitespace.
    Returns (start_char, end_char) in content if exactly one match, else None.
    """
    old_lines = old.splitlines()
    if not old_lines:
        return None
    old_stripped = [l.strip() for l in old_lines]

    # keepends=True so char offsets are exact
    content_lines = content.splitlines(keepends=True)
    content_stripped = [l.strip() for l in content_lines]

    n = len(old_stripped)
    matches = []
    for i in range(len(content_stripped) - n + 1):
        if content_stripped[i:i + n] == old_stripped:
            matches.append(i)

    if len(matches) != 1:
        return None

    i = matches[0]
    start = sum(len(l) for l in content_lines[:i])
    end = sum(len(l) for l in content_lines[:i + n])
    return start, end


def edit_file(path: str, old: str, new: str) -> str:
    """Edit a file by replacing old string with new string."""
    if not old or not old.strip():
        return "Error: 'old' string cannot be empty. To insert new content, include a unique nearby line in 'old' and add your new content in 'new'."

    if old == new:
        return "Error: 'old' and 'new' are identical. No changes would be made. If you want to insert code, include the anchor line in 'old' and the anchor plus new code in 'new'."

    try:
        path = os.path.expanduser(path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # --- Exact match ---
        if old in content:
            count = content.count(old)
            if count > 1:
                # Show the line number of each occurrence so the model can add
                # surrounding context to make the match unique
                lines = content.splitlines()
                old_first_line = old.splitlines()[0]
                hit_lines = [
                    i + 1 for i, l in enumerate(lines) if old_first_line in l
                ]
                hits_str = ", ".join(f"line {n}" for n in hit_lines[:5])
                return (
                    f"Error: Found {count} occurrences of that text ({hits_str}). "
                    f"Use read_file with start_line/end_line around one of those lines, "
                    f"then retry edit_file with enough surrounding context to be unique."
                )
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content.replace(old, new, 1))
            return f"Successfully edited {path}"

        # --- Whitespace-normalized fallback ---
        match = _find_normalized(content, old)
        if match:
            start, end = match
            new_content = content[:start] + new + content[end:]
            with open(path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return f"Successfully edited {path} (matched with normalized whitespace)"

        # --- Not found: show nearby context to help the model self-correct ---
        hint_line = next((l.strip() for l in old.splitlines() if l.strip()), "")
        if hint_line:
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if hint_line in line:
                    lo = max(0, i - 3)
                    hi = min(len(lines), i + 4)
                    context = "\n".join(f"{lo + j + 1}: {lines[lo + j]}" for j in range(hi - lo))
                    return (
                        f"Error: Exact text not found in {path}. "
                        f"Found a similar line at line {i + 1}. Nearby content:\n{context}\n"
                        f"Call read_file to get the exact text, then retry."
                    )

        return (
            f"Error: Could not find the specified text in {path}. "
            f"Call read_file to get the exact content, then retry."
        )

    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error editing file: {e}"


_BLOCKED_SHELL_PATTERNS = [
    r'\bstart\b', r'\bxdg-open\b', r'\bopen\b',
    r'\bchrome\b', r'\bfirefox\b', r'\bmsedge\b', r'\biexplore\b', r'\bbrave\b',
    r'\belectron\b', r'\bplaywright\b', r'\bpuppeteer\b', r'\bselenium\b',
    r'\becho\b',
]
_BLOCKED_SHELL_RE = re.compile('|'.join(_BLOCKED_SHELL_PATTERNS), re.IGNORECASE)


def shell(command: str) -> str:
    """Execute a shell command and return output."""
    if _BLOCKED_SHELL_RE.search(command):
        return (
            "Blocked: browser/GUI commands are not allowed. "
            "Do NOT retry this. Use a code-only alternative — "
            "draw with canvas, generate programmatic output, or skip visual testing."
        )
    try:
        _win_flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            **_win_flags
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\n(Exit code: {result.returncode})"
        if not output:
            return "(No output)"
        if len(output) > 3000:
            output = output[:3000] + "\n\n[Truncated]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds"
    except Exception as e:
        return f"Error executing command: {e}"


class _HTMLToText(HTMLParser):
    """Extract readable text from HTML, skipping scripts/styles."""

    SKIP_TAGS = {'script', 'style', 'noscript', 'svg', 'head'}
    BLOCK_TAGS = {'p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                  'li', 'tr', 'blockquote', 'section', 'article', 'header',
                  'footer', 'nav', 'pre', 'hr'}

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        elif tag in self.BLOCK_TAGS and not self._skip_depth:
            self._parts.append('\n')

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self.BLOCK_TAGS and not self._skip_depth:
            self._parts.append('\n')

    def handle_data(self, data):
        if not self._skip_depth:
            self._parts.append(data)

    def get_text(self):
        text = ''.join(self._parts)
        # Collapse runs of whitespace/blank lines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


def _html_to_text(raw_html: str) -> str:
    """Convert HTML to readable plain text."""
    parser = _HTMLToText()
    parser.feed(raw_html)
    return parser.get_text()


def _fetch_url(url: str, max_bytes: int = 200_000) -> str:
    """Shared URL fetcher. Returns raw response body."""
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read(max_bytes).decode('utf-8', errors='replace')


_CAPTCHA_SIGNALS = [
    "captcha", "robot", "are you human", "bot check", "bot detection",
    "access denied", "unusual traffic", "verify you are", "i'm not a robot",
    "please verify", "security check", "challenge-form", "cf-challenge",
]


def _is_captcha(text: str) -> bool:
    """Return True if the page looks like a bot/captcha block."""
    low = text.lower()
    return sum(1 for s in _CAPTCHA_SIGNALS if s in low) >= 2


def web_fetch(url: str) -> str:
    """Fetch a URL and return clean text (HTML stripped)."""
    try:
        raw = _fetch_url(url)
        if _is_captcha(raw):
            return "(Captcha or bot-check blocked this page — skip and use your own knowledge)"
        text = _html_to_text(raw)
        if len(text) > 3000:
            text = text[:3000] + "\n\n[Truncated]"
        return text if text else "(Page returned no readable text content)"
    except urllib.error.HTTPError as e:
        return f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}"
    except Exception as e:
        return f"Error fetching URL: {e}"


def web_search(query: str) -> str:
    """Search DuckDuckGo and return clean results."""
    try:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
        raw = _fetch_url(url)

        if _is_captcha(raw):
            return "(Captcha or bot-check blocked the search — skip and use your own knowledge)"

        # Parse DuckDuckGo Lite results
        results = []
        # DDG lite: href in double quotes, class in single quotes, class after href
        link_pattern = re.compile(
            r'''<a[^>]*href=["']([^"']*?)["'][^>]*class=['"]result-link['"][^>]*>(.*?)</a>''',
            re.DOTALL
        )
        snippet_pattern = re.compile(
            r'''<td\s+class=['"]result-snippet['"]>(.*?)</td>''',
            re.DOTALL
        )

        links = link_pattern.findall(raw)
        snippets = snippet_pattern.findall(raw)

        for i, (href, title_html) in enumerate(links):
            title = html.unescape(re.sub(r'<[^>]+>', '', title_html).strip())[:80]
            # Extract actual URL from DDG redirect
            uddg = re.search(r'uddg=([^&]+)', href)
            clean_url = urllib.parse.unquote(uddg.group(1)) if uddg else href
            snippet = ''
            if i < len(snippets):
                snippet = html.unescape(re.sub(r'<[^>]+>', '', snippets[i]).strip())[:150]
            if title:
                entry = f"{i+1}. {title}\n   {clean_url}"
                if snippet:
                    entry += f"\n   {snippet}"
                results.append(entry)

        if results:
            output = f"Search results for: {query}\n\n" + "\n\n".join(results[:5])
            return output[:2000]

        # Fallback: just extract all text if pattern didn't match
        text = _html_to_text(raw)
        if len(text) > 1500:
            text = text[:1500] + "\n\n[Truncated]"
        output = f"Search results for: {query}\n\n{text}" if text else "No results found."
        return output[:2000]

    except Exception as e:
        return f"Error searching: {e}"


def list_dir(path: str = ".") -> str:
    """List directory contents with file type indicators."""
    try:
        path = os.path.expanduser(path or ".")
        entries = sorted(os.listdir(path))
        if not entries:
            return "(Empty directory)"

        lines = []
        for entry in entries[:50]:
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                lines.append(f"  {entry}/")
            else:
                size = os.path.getsize(full)
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                lines.append(f"  {entry}  ({size_str})")

        result = f"{path}/\n" + "\n".join(lines)
        if len(entries) > 50:
            result += f"\n\n(Showing 50/{len(entries)} entries)"
        return result
    except FileNotFoundError:
        return f"Error: Directory not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def find_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern recursively."""
    try:
        path = os.path.expanduser(path or ".")
        matches = []
        for root, dirs, files in os.walk(path):
            # Skip hidden dirs and common noise
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', '.git')]
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    matches.append(os.path.join(root, name))
                    if len(matches) >= 30:
                        return "\n".join(matches) + "\n\n(Limited to 30 results)"

        if not matches:
            return f"No files matching '{pattern}' found in {path}"
        return "\n".join(matches)
    except Exception as e:
        return f"Error finding files: {e}"


def search_files(pattern: str, path: str = ".", glob: str = None) -> str:
    """Search for a text pattern inside files recursively."""
    try:
        path = os.path.expanduser(path or ".")
        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        results = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__', '.git')]
            for name in files:
                if glob and not fnmatch.fnmatch(name, glob):
                    continue
                filepath = os.path.join(root, name)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append(f"{filepath}:{i}: {line.rstrip()[:150]}")
                                if len(results) >= 30:
                                    return "\n".join(results) + "\n\n(Limited to 30 matches)"
                except (PermissionError, IsADirectoryError, OSError):
                    continue

        if not results:
            return f"No matches for '{pattern}' in {path}"
        return "\n".join(results)
    except Exception as e:
        return f"Error searching files: {e}"


# --- Time tool ---

def get_current_time() -> str:
    """Return the current local date and time."""
    from datetime import datetime
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")


# --- Memory tools ---

_MEMORY_MAX_LINES = 100


def read_memory() -> str:
    """Read the persistent MEMORY.md file."""
    path = os.path.join(AGENT_DATA_DIR, "MEMORY.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "(MEMORY.md does not exist yet — use update_memory to create it)"
    except Exception as e:
        return f"Error reading memory: {e}"


def update_memory(content: str) -> str:
    """Overwrite MEMORY.md with new content (capped at _MEMORY_MAX_LINES lines)."""
    if not content or not content.strip():
        return "Error: content cannot be empty."
    lines = content.splitlines()
    if len(lines) > _MEMORY_MAX_LINES:
        content = "\n".join(lines[:_MEMORY_MAX_LINES]) + f"\n\n[Truncated to {_MEMORY_MAX_LINES} lines]"
    path = os.path.join(AGENT_DATA_DIR, "MEMORY.md")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Memory updated ({len(content)} chars)"
    except Exception as e:
        return f"Error writing memory: {e}"


# --- Daily log tools ---

_DAILY_LOG_PATH = os.path.join(AGENT_DATA_DIR, "DAILY_LOG.md")
_daily_log_lock = threading.Lock()


def read_daily_log() -> str:
    """Read the current daily log (entries since the last 4 AM reset)."""
    try:
        with open(_DAILY_LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        return content if content.strip() else "(Daily log is empty)"
    except FileNotFoundError:
        return "(Daily log is empty)"
    except Exception as e:
        return f"Error reading daily log: {e}"


def append_daily_log(entry: str) -> str:
    """Append a timestamped entry to the daily log. Use this to note anything worth remembering from the current conversation."""
    from datetime import datetime
    if not entry or not entry.strip():
        return "Error: entry cannot be empty."
    ts = datetime.now().strftime("%H:%M")
    line = f"[{ts}] {entry.strip()}\n"
    try:
        with _daily_log_lock:
            with open(_DAILY_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
        return "Logged."
    except Exception as e:
        return f"Error writing daily log: {e}"


def clear_context() -> str:
    """Signal that conversation context should be cleared on the next user interaction. Used by the 4 AM daily reset."""
    flag_path = os.path.join(AGENT_DATA_DIR, "CLEAR_CONTEXT")
    try:
        with open(flag_path, "w", encoding="utf-8") as f:
            from datetime import datetime
            f.write(datetime.now().isoformat())
        return "Context clear flagged — will take effect on next interaction."
    except Exception as e:
        return f"Error writing clear flag: {e}"


# --- Schedule tools ---

_SCHEDULE_HEADER = (
    "# Stoat Schedule\n"
    "#\n"
    "# Formats:\n"
    "#   YYYY-MM-DD HH:MM  | task description     (one-time)\n"
    "#   daily HH:MM       | task description     (repeats daily)\n"
    "#   every Xh          | task description     (repeats every X hours)\n"
    "#   every Xm          | task description     (repeats every X minutes)\n"
    "#\n"
    "# Do not hand-edit [next: ...] or [done: ...] annotations.\n"
)


def read_schedule() -> str:
    """Read the persistent SCHEDULE.md file."""
    path = os.path.join(AGENT_DATA_DIR, "SCHEDULE.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "(SCHEDULE.md is empty — use update_schedule to add tasks)"
    except Exception as e:
        return f"Error reading schedule: {e}"


def update_schedule(content: str) -> str:
    """Overwrite SCHEDULE.md with new content. Preserve the header comment block."""
    if not content or not content.strip():
        return "Error: content cannot be empty."
    # Ensure header is present
    if not content.lstrip().startswith("#"):
        content = _SCHEDULE_HEADER + "\n" + content
    path = os.path.join(AGENT_DATA_DIR, "SCHEDULE.md")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Schedule updated ({content.count(chr(10))+1} lines)"
    except Exception as e:
        return f"Error writing schedule: {e}"


# --- Tool Permission System ---

# Tools that default to "ask" (None) — potentially destructive operations
DANGEROUS_TOOLS = {"write_file", "edit_file", "shell"}

# All tool names (derived from TOOL_FUNCTIONS below at module init)
ALL_TOOL_NAMES: set[str] = set()

# None = ask each time, True = always allow, False = always deny
# Dangerous tools default to None (ask); safe tools default to True (allowed).
_tool_permissions: dict[str, bool | None] = {}
_permissions_lock = threading.Lock()  # protects _tool_permissions across threads


def _init_permissions() -> None:
    """Initialize permissions for all tools (called after TOOL_FUNCTIONS is defined)."""
    ALL_TOOL_NAMES.update(TOOL_FUNCTIONS.keys())
    for name in ALL_TOOL_NAMES:
        _tool_permissions[name] = None if name in DANGEROUS_TOOLS else True


def is_dangerous(name: str) -> bool:
    """Check if a tool defaults to requiring permission (write_file, edit_file, shell)."""
    return name in DANGEROUS_TOOLS


def get_permission(name: str) -> bool | None:
    """Get the current permission setting for a tool."""
    with _permissions_lock:
        return _tool_permissions.get(name)


def set_permission(name: str, value: bool | None) -> None:
    """Set a tool's permission (True=allow, False=deny, None=ask)."""
    with _permissions_lock:
        if name in _tool_permissions:
            _tool_permissions[name] = value


def get_all_permissions() -> dict[str, bool | None]:
    """Return a copy of all tool permissions."""
    with _permissions_lock:
        return dict(_tool_permissions)


# Map tool names to functions
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "shell": shell,
    "web_fetch": web_fetch,
    "web_search": web_search,
    "list_dir": list_dir,
    "find_files": find_files,
    "search_files": search_files,
    "read_schedule": read_schedule,
    "update_schedule": update_schedule,
    "get_current_time": get_current_time,
    "read_memory": read_memory,
    "update_memory": update_memory,
    "read_daily_log": read_daily_log,
    "append_daily_log": append_daily_log,
    "clear_context": clear_context,
}


_init_permissions()


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with the given arguments."""
    if name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{name}'"

    try:
        func = TOOL_FUNCTIONS[name]
        return func(**arguments)
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error executing {name}: {e}"
