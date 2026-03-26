"""LLM client — OpenAI-compatible API wrapper for llama-server.

Handles streaming chat completions, tool call accumulation, token tracking,
and keyboard interrupt (Esc to cancel generation). Uses only stdlib (urllib)
so the project has zero pip dependencies.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import sys
import threading
import time
import urllib.request
import urllib.error

from config import TEMPERATURE, NUM_CTX, MODEL_FILE
from llm_server import SERVER_HOST, get_vram_info

# Platform-specific imports for keyboard handling
IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import msvcrt
else:
    import select
    import termios
    import tty


class KeyboardMonitor:
    """Non-blocking Esc key detection for interrupting LLM generation.

    On Windows, uses msvcrt.kbhit()/getch(). On Unix, switches stdin to
    cbreak mode and polls with select(). Restores terminal state on exit.
    """

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
        self.model = MODEL_FILE
        self.preset = "basic_compacted"


# Global settings
settings = Settings()


class TokenUsage:
    """Track token usage across the conversation."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def update(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        if prompt_tokens:
            self.prompt_tokens = prompt_tokens
        if completion_tokens:
            self.completion_tokens = completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def percentage(self) -> float:
        if settings.context_size == 0:
            return 0.0
        return (self.total_tokens / settings.context_size) * 100

    def format_status(self) -> str:
        pct = self.percentage
        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        if pct < 50:
            color = "\033[32m"
        elif pct < 80:
            color = "\033[33m"
        else:
            color = "\033[31m"

        reset = "\033[0m"
        return f"{color}{bar} {pct:.1f}% ({self.total_tokens}/{settings.context_size}){reset}"


# Global token tracker
token_usage = TokenUsage()


def load_image_b64(path: str) -> tuple[str, str]:
    """Load an image file and return (mime_type, base64_data).

    Raises FileNotFoundError if the path doesn't exist.
    """
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return mime, data


def _get_api_url() -> str:
    """Get the llama-server API base URL (re-reads in case port changed)."""
    from llm_server import SERVER_HOST
    return SERVER_HOST  # may change after setup_environment(port=...)


def _api_get(path: str, timeout: int = 10) -> dict:
    """Make a GET request to the llama-server API. Returns parsed JSON or {}."""
    url = f"{_get_api_url()}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


def _convert_tools_to_openai(tools: list[dict]) -> list[dict]:
    """Ensure tools are in OpenAI function-calling format.

    llama-server uses the OpenAI schema, so this is mostly a pass-through.
    But some Stoat tools may be in the short format — normalize them.
    """
    result = []
    for tool in tools:
        if "type" in tool and tool["type"] == "function":
            result.append(tool)
        elif "function" in tool:
            result.append({"type": "function", **tool})
        elif "name" in tool:
            # Short format: {name, description, parameters}
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                }
            })
        else:
            result.append(tool)
    return result


def _convert_tool_calls_from_openai(choices_tc: list) -> list[dict]:
    """Convert OpenAI-format tool_calls to the flat format Stoat expects.

    OpenAI format:  [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
    Stoat expects:  [{"function": {"name": "...", "arguments": {...}}}]
    """
    result = []
    for tc in choices_tc:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_raw = func.get("arguments", "{}")
        # Arguments may be a JSON string
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"raw": args_raw}
        else:
            args = args_raw
        result.append({"function": {"name": name, "arguments": args}})
    return result


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Ensure messages are in valid OpenAI format for llama-server with --jinja.

    Ensures tool_calls have proper 'id' fields and tool results reference them,
    as required by the OpenAI chat format.
    """
    result = []
    call_counter = 0
    for msg in messages:
        role = msg.get("role", "user")
        if role == "assistant" and "tool_calls" in msg:
            # Ensure each tool_call has an id
            sanitized_tcs = []
            for tc in msg["tool_calls"]:
                call_counter += 1
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, dict):
                    args_str = json.dumps(args)
                else:
                    args_str = str(args)
                sanitized_tcs.append({
                    "id": f"call_{call_counter}",
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": args_str,
                    },
                })
            result.append({
                "role": "assistant",
                "content": msg.get("content") or None,
                "tool_calls": sanitized_tcs,
            })
        elif role == "tool":
            # Ensure tool result references the correct call id
            result.append({
                "role": "tool",
                "content": msg.get("content", ""),
                "tool_call_id": msg.get("tool_call_id", f"call_{call_counter}"),
            })
        else:
            result.append(msg)
    return result


def chat_stream(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    timeout: int = 300,
    print_output: bool = True,
    color: str = "\033[36m",
    output_callback: "Callable[[str], None] | None" = None,
) -> dict:
    """Send a streaming chat request to llama-server (OpenAI-compatible API).

    Returns a dict with 'message' containing 'role', 'content',
    and optionally 'tool_calls'. Uses urllib for zero-dependency streaming.
    """
    url = f"{_get_api_url()}/v1/chat/completions"

    payload = {
        "model": model or settings.model,
        "messages": _sanitize_messages(messages),
        "stream": True,
        "temperature": settings.temperature,
        "stream_options": {"include_usage": True},
    }

    if tools:
        payload["tools"] = _convert_tools_to_openai(tools)

    # Accumulated response
    full_content = ""
    tool_calls_accum = {}  # index -> {name, arguments_str}
    first_token_received = False

    # Think-tag handling: buffer early content to detect <think> even when
    # streamed character-by-character. States: "detecting" -> "thinking" | "streaming"
    _THINK_OPEN = "<think>"
    _THINK_CLOSE = "</think>"
    _think_state = "detecting"  # "detecting", "thinking", "streaming"
    _think_buffer = ""          # Accumulates tokens during detecting/thinking phases

    interrupted = False

    # Thinking spinner — shows animated indicator until first token arrives
    _SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    _spinner_active = [True]   # mutable flag shared with spinner thread
    _spinner_thread = [None]   # holds Thread reference so _stop_spinner can join it
    _spinner_start = [time.time()]

    def _thinking_spinner():
        i = 0
        while _spinner_active[0]:
            elapsed = time.time() - _spinner_start[0]
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            sys.stdout.write(f"\r\033[2m  {frame} Thinking... {elapsed:.0f}s\033[0m")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1

    def _stop_spinner():
        """Stop the thinking spinner and clear its line."""
        if _spinner_active[0]:
            _spinner_active[0] = False
            t = _spinner_thread[0]
            if t is not None and t.is_alive():
                t.join(timeout=0.5)  # wait for spinner to finish its current write
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        if print_output:
            _spinner_start[0] = time.time()
            spinner_thread = threading.Thread(target=_thinking_spinner, daemon=True)
            _spinner_thread[0] = spinner_thread
            spinner_thread.start()

        with KeyboardMonitor() as kb:
            response = urllib.request.urlopen(req, timeout=timeout)

            # Read SSE stream line by line
            for raw_line in response:
                if kb.escape_pressed():
                    interrupted = True
                    break

                line_str = raw_line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                # SSE format: "data: {...}" or "data: [DONE]"
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]

                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Token usage (in the final chunk — choices may be empty)
                usage = chunk.get("usage")
                if usage:
                    token_usage.update(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                    )

                # Extract from OpenAI SSE chunk
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})

                # Content tokens — handle <think>...</think> blocks from reasoning models.
                # Tags may arrive character-by-character, so we buffer early tokens
                # and check the accumulated buffer to detect them reliably.
                content = delta.get("content", "")
                if content:
                    if _think_state == "detecting":
                        # Buffer tokens until we can confirm or rule out <think>
                        _think_buffer += content
                        if _THINK_OPEN in _think_buffer:
                            # Confirmed: model is reasoning — keep spinner, enter thinking state
                            _think_state = "thinking"
                            # Check if </think> already arrived in same buffer
                            if _THINK_CLOSE in _think_buffer:
                                after = _think_buffer.split(_THINK_CLOSE, 1)[1]
                                _think_buffer = ""
                                _think_state = "streaming"
                                if after:
                                    first_token_received = True
                                    _stop_spinner()
                                    if print_output:
                                        sys.stdout.write(color + after)
                                        sys.stdout.flush()
                                    if output_callback:
                                        output_callback(after)
                                    full_content += after
                        elif not _THINK_OPEN.startswith(_think_buffer.lstrip()):
                            # Buffer can't become <think> — flush as real content
                            _think_state = "streaming"
                            first_token_received = True
                            _stop_spinner()
                            if print_output:
                                sys.stdout.write(color + _think_buffer)
                                sys.stdout.flush()
                            if output_callback:
                                output_callback(_think_buffer)
                            full_content += _think_buffer
                            _think_buffer = ""

                    elif _think_state == "thinking":
                        # Inside <think> block — buffer silently, keep spinner running
                        _think_buffer += content
                        if _THINK_CLOSE in _think_buffer:
                            after = _think_buffer.split(_THINK_CLOSE, 1)[1]
                            _think_buffer = ""
                            _think_state = "streaming"
                            if after:
                                first_token_received = True
                                _stop_spinner()
                                if print_output:
                                    sys.stdout.write(color + after)
                                    sys.stdout.flush()
                                if output_callback:
                                    output_callback(after)
                                full_content += after

                    else:  # "streaming" — normal output
                        if not first_token_received:
                            first_token_received = True
                            _stop_spinner()
                            if print_output:
                                sys.stdout.write(color)
                                sys.stdout.flush()
                        if print_output:
                            sys.stdout.write(content)
                            sys.stdout.flush()
                        if output_callback:
                            output_callback(content)
                        full_content += content

                # Tool calls (streamed incrementally)
                for tc in delta.get("tool_calls", []):
                    if not first_token_received:
                        first_token_received = True
                        _stop_spinner()
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": "",
                        }
                    func = tc.get("function", {})
                    if "name" in func and func["name"]:
                        tool_calls_accum[idx]["name"] = func["name"]
                    if "arguments" in func:
                        tool_calls_accum[idx]["arguments"] += func["arguments"]

        _stop_spinner()

        if print_output and first_token_received:
            sys.stdout.write("\033[0m\n")
            sys.stdout.flush()

        if interrupted:
            return {
                "message": {
                    "role": "assistant",
                    "content": full_content,
                },
                "interrupted": True,
            }

        # Build result
        result = {
            "message": {
                "role": "assistant",
                "content": full_content,
            }
        }

        # Convert accumulated tool calls to Stoat format
        if tool_calls_accum:
            openai_tcs = []
            for idx in sorted(tool_calls_accum):
                tc = tool_calls_accum[idx]
                openai_tcs.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                })
            result["message"]["tool_calls"] = _convert_tool_calls_from_openai(openai_tcs)

        return result

    except urllib.error.URLError as e:
        _stop_spinner()
        if isinstance(e.reason, socket.timeout):
            return {
                "message": {
                    "role": "assistant",
                    "content": f"Error: Request timed out after {timeout} seconds.",
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": f"Error: Could not connect to llama-server at {_get_api_url()}. Is the server running?",
            }
        }
    except urllib.error.HTTPError as e:
        _stop_spinner()
        return {
            "message": {
                "role": "assistant",
                "content": f"HTTP Error from llama-server: {e.code} {e.reason}",
            }
        }
    except socket.timeout:
        _stop_spinner()
        return {
            "message": {
                "role": "assistant",
                "content": f"Error: Request timed out after {timeout} seconds.",
            }
        }


def list_models() -> list[str]:
    """List available models from llama-server."""
    data = _api_get("/v1/models")
    return [m["id"] for m in data.get("data", [])]


def get_loaded_models() -> list[str]:
    """Get models currently loaded. Same as list_models for llama-server."""
    return list_models()


def get_model_info(model: str = None) -> dict:
    """Get model metadata — reads from GGUF file for accurate values.

    Falls back to llama-server /props if GGUF read fails.
    """
    from llm_server import get_gguf_metadata, find_model

    # Try reading metadata directly from the GGUF file
    model_file = model or settings.model
    model_path = find_model(model_file)
    if model_path:
        meta = get_gguf_metadata(model_path)
        if meta and meta.get("context_length"):
            return {
                "context_length": meta.get("context_length"),
                "parameter_count": meta.get("parameter_count"),
                "block_count": meta.get("block_count"),
                "head_count_kv": meta.get("head_count_kv"),
                "head_count": meta.get("head_count"),
                "embedding_length": meta.get("embedding_length"),
                "quantization": meta.get("quantization", ""),
                "architecture": meta.get("architecture", ""),
                "model_name": meta.get("model_name", ""),
            }

    # Fallback: query llama-server
    data = _api_get("/props")
    if data:
        return {
            "context_length": data.get("default_generation_settings", {}).get("n_ctx"),
            "parameter_count": None,
            "block_count": None,
            "head_count_kv": None,
            "head_count": None,
            "embedding_length": None,
            "quantization": "",
        }
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


def estimate_recommended_ctx(model_info: dict, vram_info: dict, model_loaded: bool = False, model_file_path: str = None) -> dict:
    """Estimate recommended context window size based on available VRAM."""
    if not vram_info or not model_info:
        return {}

    param_count = model_info.get("parameter_count")

    if param_count:
        quant = model_info.get("quantization", "").lower().replace(" ", "")
        bytes_per_param = _QUANT_BYTES.get(quant, 2.0)
        model_vram_mb = (param_count * bytes_per_param) / (1024 ** 2)
    elif model_file_path and os.path.isfile(model_file_path):
        # For quantized GGUFs, file size ≈ VRAM usage (weights are mmap'd)
        model_vram_mb = os.path.getsize(model_file_path) / (1024 ** 2)
    else:
        return {}

    OVERHEAD_MB = 512
    if model_loaded:
        free_for_kv_mb = vram_info["free_mb"] - OVERHEAD_MB
    else:
        # Model isn't loaded yet — base KV budget on total VRAM, not current free
        # (free may be low due to OS/other apps that will yield once the model loads)
        free_for_kv_mb = vram_info["total_mb"] - model_vram_mb - OVERHEAD_MB

    if free_for_kv_mb <= 0:
        return {
            "model_vram_mb": model_vram_mb,
            "free_for_kv_mb": 0,
            "recommended_ctx": None,
        }

    block_count = model_info.get("block_count")
    head_count_kv = model_info.get("head_count_kv")
    head_count = model_info.get("head_count")
    embedding_length = model_info.get("embedding_length")

    architecture_known = all(x for x in [block_count, head_count_kv, head_count, embedding_length])

    if architecture_known:
        head_dim = embedding_length // head_count
        kv_bytes_per_token = 2 * block_count * head_count_kv * head_dim * 2
    else:
        kv_bytes_per_token = int(200 * (param_count / 1e9))

    vram_ctx = int((free_for_kv_mb * 1024 ** 2) / kv_bytes_per_token)
    vram_ctx = (vram_ctx // 1024) * 1024

    model_max_ctx = model_info.get("context_length")
    recommended_ctx = min(vram_ctx, model_max_ctx) if model_max_ctx else vram_ctx

    return {
        "model_vram_mb": model_vram_mb,
        "free_for_kv_mb": free_for_kv_mb,
        "kv_bytes_per_token": kv_bytes_per_token,
        "recommended_ctx": max(0, recommended_ctx),
        "vram_ctx": max(0, vram_ctx),
        "model_max_ctx": model_max_ctx,
        "architecture_known": architecture_known,
        "model_loaded": model_loaded,
    }


def check_connection() -> bool:
    """Check if llama-server is reachable and healthy."""
    data = _api_get("/health", timeout=5)
    return data.get("status") == "ok"


def get_token_usage() -> TokenUsage:
    return token_usage


def get_settings() -> Settings:
    return settings


def reset_token_usage() -> None:
    global token_usage
    token_usage = TokenUsage()
