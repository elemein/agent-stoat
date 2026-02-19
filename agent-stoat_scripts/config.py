"""Configuration for the local LLM agent."""

import os
import platform
import subprocess


def _is_wsl() -> bool:
    """Check if running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (FileNotFoundError, PermissionError):
        return False


def get_ollama_host() -> str:
    """Get the Ollama host URL, works on Windows, WSL, and native Linux."""
    if platform.system() == "Windows":
        return "http://localhost:11434"

    if _is_wsl():
        # WSL: Ollama runs on the Windows host, reach it via gateway IP
        try:
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True
            )
            parts = result.stdout.strip().split()
            if len(parts) >= 3 and parts[0] == "default" and parts[1] == "via":
                return f"http://{parts[2]}:11434"
        except Exception:
            pass

    # Native Linux or fallback
    return "http://localhost:11434"


# Ollama API endpoint (auto-detected based on environment)
OLLAMA_HOST = get_ollama_host()

# Model to use
MODEL = "qwen2.5-coder:14b-instruct-q5_K_M"

# Generation settings
TEMPERATURE = 0.7
NUM_CTX = 8192  # Context window size

# Auto-compaction thresholds (percentage of context window)
COMPACT_THRESHOLD = 70   # Auto-compact at this % of context
COMPACT_EMERGENCY = 85   # Force compact at this %

# Max tool-calling iterations per user message (None = unlimited)
MAX_ITERATIONS = 20

# System prompt — loaded from prompt.md in this directory
_PLATFORM = platform.system()
_SHELL_INFO = "PowerShell/cmd on Windows" if _PLATFORM == "Windows" else "bash on Linux"

_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.md")

try:
    with open(_PROMPT_FILE, "r", encoding="utf-8") as _f:
        SYSTEM_PROMPT = _f.read().strip().format(platform=_PLATFORM, shell=_SHELL_INFO)
except FileNotFoundError:
    raise FileNotFoundError(
        f"System prompt file not found: {_PROMPT_FILE}\n"
        "Expected prompt.md in agent-stoat_scripts/."
    )
