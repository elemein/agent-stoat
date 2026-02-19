"""Ollama API client."""

import json
import subprocess
import sys
import requests
from typing import Optional

from config import OLLAMA_HOST, MODEL, TEMPERATURE, NUM_CTX, MAX_ITERATIONS

# Platform-specific imports for keyboard handling
IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import msvcrt
else:
    import select
    import termios
    import tty


class KeyboardMonitor:
    """Context manager for non-blocking keyboard input. Works on both Windows and Unix."""

    def __init__(self):
        self.old_settings = None
        self.active = False

    def __enter__(self):
        if IS_WINDOWS:
            self.active = True
        else:
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                self.active = True
            except (termios.error, AttributeError):
                pass
        return self

    def __exit__(self, *args):
        if not IS_WINDOWS and self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        self.active = False

    def escape_pressed(self) -> bool:
        """Check if Esc was pressed (non-blocking)."""
        if not self.active:
            return False
        if IS_WINDOWS:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch == b'\x1b':
                    return True
        else:
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    return True
        return False


class Settings:
    """Runtime settings that can be changed dynamically."""

    def __init__(self):
        self.context_size = NUM_CTX
        self.temperature = TEMPERATURE
        self.model = MODEL
        self.max_iterations = MAX_ITERATIONS  # None = unlimited


# Global settings
settings = Settings()


class TokenUsage:
    """Track token usage across the conversation."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def update(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """Update token counts."""
        if prompt_tokens:
            self.prompt_tokens = prompt_tokens
        if completion_tokens:
            self.completion_tokens = completion_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt is what matters for context)."""
        return self.prompt_tokens

    @property
    def percentage(self) -> float:
        """Percentage of context window used."""
        if settings.context_size == 0:
            return 0.0
        return (self.total_tokens / settings.context_size) * 100

    def format_status(self) -> str:
        """Format a status string showing token usage."""
        pct = self.percentage
        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Color based on usage
        if pct < 50:
            color = "\033[32m"  # Green
        elif pct < 80:
            color = "\033[33m"  # Yellow
        else:
            color = "\033[31m"  # Red

        reset = "\033[0m"
        return f"{color}{bar} {pct:.1f}% ({self.total_tokens}/{settings.context_size}){reset}"


# Global token tracker
token_usage = TokenUsage()


def chat_stream(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: Optional[str] = None,
    timeout: int = 300,
    print_output: bool = True
) -> dict:
    """
    Send a streaming chat request to Ollama.
    Prints tokens as they arrive and returns the complete response.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: Optional list of tool definitions
        model: Model name to use (defaults to settings.model)
        timeout: Request timeout in seconds (default 300 = 5 minutes)
        print_output: Whether to print streamed content (default True)

    Returns:
        Complete Ollama response dict with 'message' containing 'role', 'content',
        and optionally 'tool_calls'
    """
    url = f"{OLLAMA_HOST}/api/chat"

    payload = {
        "model": model or settings.model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": settings.temperature,
            "num_ctx": settings.context_size,
        }
    }

    if tools:
        payload["tools"] = tools

    # Accumulated response
    full_content = ""
    tool_calls = []

    # Show initial indicator
    if print_output:
        sys.stdout.write("\033[36m")  # Cyan color
        sys.stdout.flush()

    interrupted = False

    try:
        with KeyboardMonitor() as kb:
            response = requests.post(url, json=payload, timeout=timeout, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                # Check for Esc key
                if kb.escape_pressed():
                    interrupted = True
                    break

                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract content from this chunk
                message = chunk.get("message", {})
                content = message.get("content", "")

                if content and print_output:
                    sys.stdout.write(content)
                    sys.stdout.flush()

                full_content += content

                # Check for tool calls (usually in final chunk)
                if "tool_calls" in message and message["tool_calls"]:
                    tool_calls = message["tool_calls"]

                # Check if done - update token counts
                if chunk.get("done"):
                    token_usage.update(
                        prompt_tokens=chunk.get("prompt_eval_count", 0),
                        completion_tokens=chunk.get("eval_count", 0)
                    )

        if print_output:
            sys.stdout.write("\033[0m\n")  # Reset color, newline
            sys.stdout.flush()

        if interrupted:
            # Return partial response instead of raising
            # This allows the caller to see what was generated before interrupt
            return {
                "message": {
                    "role": "assistant",
                    "content": full_content,
                },
                "interrupted": True
            }

        # Build complete response object
        result = {
            "message": {
                "role": "assistant",
                "content": full_content,
            }
        }
        if tool_calls:
            result["message"]["tool_calls"] = tool_calls

        return result

    except requests.exceptions.ConnectionError:
        if print_output:
            sys.stdout.write("\033[0m\n")
            sys.stdout.flush()
        return {
            "message": {
                "role": "assistant",
                "content": f"Error: Could not connect to Ollama at {OLLAMA_HOST}. Is Ollama running?"
            }
        }
    except requests.exceptions.Timeout:
        if print_output:
            sys.stdout.write("\033[0m\n")
            sys.stdout.flush()
        return {
            "message": {
                "role": "assistant",
                "content": f"Error: Request to Ollama timed out after {timeout} seconds."
            }
        }
    except requests.exceptions.HTTPError as e:
        if print_output:
            sys.stdout.write("\033[0m\n")
            sys.stdout.flush()
        return {
            "message": {
                "role": "assistant",
                "content": f"HTTP Error from Ollama: {e}"
            }
        }


def chat(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: Optional[str] = None,
    timeout: int = 300
) -> dict:
    """Send a chat request with streaming output."""
    return chat_stream(messages, tools, model, timeout, print_output=True)


def list_models() -> list[str]:
    """Get list of available models from Ollama."""
    url = f"{OLLAMA_HOST}/api/tags"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def get_loaded_models() -> list[str]:
    """Get list of models currently loaded in Ollama's VRAM (/api/ps)."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def get_model_info(model: str = None) -> dict:
    """Get model metadata from Ollama including architecture details for VRAM estimation."""
    url = f"{OLLAMA_HOST}/api/show"
    try:
        response = requests.post(url, json={"name": model or settings.model}, timeout=10)
        response.raise_for_status()
        data = response.json()
        info = data.get("model_info", {})

        ctx = None
        params = None
        block_count = None
        head_count_kv = None
        head_count = None
        embedding_length = None

        for k, v in info.items():
            if k.endswith(".context_length"):
                ctx = v
            elif k == "general.parameter_count":
                params = v
            elif k.endswith(".block_count"):
                block_count = v
            elif k.endswith(".attention.head_count_kv"):
                head_count_kv = v
            elif k.endswith(".attention.head_count"):
                head_count = v
            elif k.endswith(".embedding_length"):
                embedding_length = v

        # Quantization level from details (e.g. "Q5_K_M")
        quant = data.get("details", {}).get("quantization_level", "")

        return {
            "context_length": ctx,
            "parameter_count": params,
            "block_count": block_count,
            "head_count_kv": head_count_kv,
            "head_count": head_count,
            "embedding_length": embedding_length,
            "quantization": quant,
        }
    except Exception:
        return {}


def get_vram_info() -> dict:
    """Query nvidia-smi for GPU VRAM. Returns {} if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return {}
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return {}
        # Sum across all GPUs
        total_mb = free_mb = 0
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 2:
                total_mb += int(parts[0].strip())
                free_mb += int(parts[1].strip())
        return {"total_mb": total_mb, "free_mb": free_mb, "gpu_count": len(lines)}
    except Exception:
        return {}


# Bytes per parameter by quantization level
_QUANT_BYTES: dict[str, float] = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0, "q8_1": 1.0,
    "q6_k": 0.75,
    "q5_0": 0.625, "q5_1": 0.6875, "q5_k_m": 0.625, "q5_k_s": 0.625,
    "q4_0": 0.5, "q4_1": 0.5625, "q4_k_m": 0.5, "q4_k_s": 0.5,
    "q3_k_m": 0.375, "q3_k_s": 0.375, "q3_k_l": 0.375,
    "q2_k": 0.25,
    "iq4_xs": 0.45, "iq3_xxs": 0.33,
}


def estimate_recommended_ctx(model_info: dict, vram_info: dict, model_loaded: bool = False) -> dict:
    """
    Estimate recommended context window size based on available VRAM and model architecture.

    Returns a dict with:
      model_vram_mb        — estimated VRAM the model weights use
      free_for_kv_mb       — VRAM available for KV cache after model + overhead
      kv_bytes_per_token   — KV cache cost per token in bytes
      recommended_ctx      — recommended max context (multiple of 1024), or None
      architecture_known   — True if exact KV formula was used, False if approximated
    """
    if not vram_info or not model_info:
        return {}

    param_count = model_info.get("parameter_count")
    if not param_count:
        return {}

    # Estimate model weight VRAM
    quant = model_info.get("quantization", "").lower().replace(" ", "")
    bytes_per_param = _QUANT_BYTES.get(quant, 2.0)  # default FP16 if unknown
    model_vram_mb = (param_count * bytes_per_param) / (1024 ** 2)

    # If the model is already loaded in VRAM, free_mb already reflects that —
    # subtracting model_vram_mb again would double-count it.
    OVERHEAD_MB = 512
    if model_loaded:
        free_for_kv_mb = vram_info["free_mb"] - OVERHEAD_MB
    else:
        free_for_kv_mb = vram_info["free_mb"] - model_vram_mb - OVERHEAD_MB

    if free_for_kv_mb <= 0:
        return {
            "model_vram_mb": model_vram_mb,
            "free_for_kv_mb": 0,
            "recommended_ctx": None,
        }

    # KV cache bytes per token
    block_count = model_info.get("block_count")
    head_count_kv = model_info.get("head_count_kv")
    head_count = model_info.get("head_count")
    embedding_length = model_info.get("embedding_length")

    architecture_known = all(x for x in [block_count, head_count_kv, head_count, embedding_length])

    if architecture_known:
        head_dim = embedding_length // head_count
        # K and V tensors, FP16 (2 bytes), per layer, per KV head, per token
        kv_bytes_per_token = 2 * block_count * head_count_kv * head_dim * 2
    else:
        # Fallback: empirical ~200 bytes per token per billion parameters
        kv_bytes_per_token = int(200 * (param_count / 1e9))

    vram_ctx = int((free_for_kv_mb * 1024 ** 2) / kv_bytes_per_token)
    vram_ctx = (vram_ctx // 1024) * 1024  # round down to nearest 1024

    # Cap at the model's own maximum context length
    model_max_ctx = model_info.get("context_length")
    recommended_ctx = min(vram_ctx, model_max_ctx) if model_max_ctx else vram_ctx

    return {
        "model_vram_mb": model_vram_mb,
        "free_for_kv_mb": free_for_kv_mb,
        "kv_bytes_per_token": kv_bytes_per_token,
        "recommended_ctx": max(0, recommended_ctx),
        "vram_ctx": max(0, vram_ctx),        # uncapped VRAM-based value
        "model_max_ctx": model_max_ctx,
        "architecture_known": architecture_known,
        "model_loaded": model_loaded,
    }


def check_connection() -> bool:
    """Check if Ollama is reachable."""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_token_usage() -> TokenUsage:
    """Get the current token usage tracker."""
    return token_usage


def get_settings() -> Settings:
    """Get the current settings."""
    return settings


def reset_token_usage() -> None:
    """Reset token usage (call when clearing history)."""
    global token_usage
    token_usage = TokenUsage()
