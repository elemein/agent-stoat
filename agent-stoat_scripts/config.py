"""Configuration defaults and prompt loading for Agent Stoat.

Values here can be overridden at runtime by .agent-stoat/settings.json
(see _apply_settings_overlay).
"""

import json
import os
import platform


# ── Model defaults ────────────────────────────────────────────────────────
# Used for auto-download and initial model selection.
MODEL_REPO = "unsloth/Qwen3.5-35B-A3B-GGUF"
MODEL_FILE = "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"

# ── Generation defaults ───────────────────────────────────────────────────
TEMPERATURE = 0.7
NUM_CTX = 32768       # Context window size in tokens
MAX_ITERATIONS = 50   # Max tool-calling rounds per user message

# ── Platform info (injected into prompts via {platform}/{shell}) ──────────
_PLATFORM = platform.system()
def _detect_shell() -> str:
    if _PLATFORM != "Windows":
        return os.environ.get("SHELL", "bash")
    if "PSModulePath" in os.environ:
        return "PowerShell"
    return "cmd.exe"

_SHELL_INFO = _detect_shell()

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_prompt(filename: str) -> str:
    """Load a prompt file from agent-stoat_scripts/, substituting {platform} and {shell}."""
    path = os.path.join(_SCRIPTS_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Only substitute if the file actually uses these placeholders
        if "{platform}" in content or "{shell}" in content:
            content = content.format(platform=_PLATFORM, shell=_SHELL_INFO)
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {path}")


def _apply_settings_overlay() -> None:
    """Override defaults from .agent-stoat/settings.json if it exists.

    Expected format: {"model": "...", "temperature": 0.7, "num_ctx": 32768}
    """
    global MODEL_FILE, TEMPERATURE, NUM_CTX
    overlay_path = os.path.join(os.path.dirname(_SCRIPTS_DIR), ".agent-stoat", "settings.json")
    if not os.path.exists(overlay_path):
        return
    try:
        with open(overlay_path, "r", encoding="utf-8") as f:
            ov = json.load(f)
        MODEL_FILE  = ov.get("model", MODEL_FILE)
        TEMPERATURE = ov.get("temperature", TEMPERATURE)
        NUM_CTX     = ov.get("num_ctx", NUM_CTX)
    except Exception:
        pass


_apply_settings_overlay()

# Default system prompt (loaded once at import time)
BASIC_PROMPT = _load_prompt("prompt_basic.md")
