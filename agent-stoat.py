#!/usr/bin/env python3
"""Stoat — local-first coding agent powered by llama.cpp."""

import json
import os
import re
import sys

# Add scripts directory to path so supporting modules can be imported
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent-stoat_scripts"))

try:
    import readline  # Enables arrow keys, history in REPL
except ImportError:
    readline = None  # Not available on Windows without pyreadline3

from config import BASIC_PROMPT, MODEL_REPO, MODEL_FILE, _load_prompt
from llm_client import (
    chat_stream, get_token_usage,
    get_settings, reset_token_usage, get_model_info,
    get_vram_info, estimate_recommended_ctx, get_loaded_models
)
from llm_server import (
    setup_environment, load_model, stop_server, is_server_running,
    list_local_models, get_models_dir, download_model, find_model,
    get_gguf_metadata, get_backend, set_backend, SERVER_HOST,
)
from tools import (
    TOOLS, execute_tool, is_dangerous, get_permission,
    set_permission, get_all_permissions, DANGEROUS_TOOLS, ALL_TOOL_NAMES,
    AGENT_DATA_DIR, WORKING_DIR
)
from chat_engine import ChatEngine


# ANSI colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def print_colored(text: str, color: str = RESET) -> None:
    print(f"{color}{text}{RESET}")


def _shorten_model_name(name: str, max_len: int = 20) -> str:
    """Shorten a GGUF model filename for display (e.g. 'Qwen3.5-35B-A3B-UD-Q4_K_XL')."""
    if not name:
        return "(none)"
    # Strip .gguf extension
    short = name.rsplit(".gguf", 1)[0] if name.lower().endswith(".gguf") else name
    if len(short) <= max_len:
        return short
    return short[:max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_PRESETS_DIR = os.path.join(AGENT_DATA_DIR, "presets")


def list_presets() -> list[str]:
    """Return sorted names of available presets (without .json extension)."""
    if not os.path.isdir(_PRESETS_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(_PRESETS_DIR)
        if f.endswith(".json")
    )


def load_preset(name: str) -> dict:
    """Load and return a preset's JSON config by name."""
    path = os.path.join(_PRESETS_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Chat engine dispatch
# ---------------------------------------------------------------------------

# Persists across turns within a session
_conv_history: list[dict] = []


def reset_conv_history() -> None:
    global _conv_history
    _conv_history = []


def _get_usage_pct() -> float:
    """Return current context usage as a percentage (0-100)."""
    return get_token_usage().percentage


def run_chat(goal: str) -> None:
    """Run the current preset's chat engine with the given goal."""
    settings = get_settings()

    try:
        preset = load_preset(settings.preset)
    except FileNotFoundError:
        print_colored(f"  Preset '{settings.preset}' not found. Falling back to basic_compacted.", RED)
        settings.preset = "basic_compacted"
        try:
            preset = load_preset(settings.preset)
        except FileNotFoundError:
            print_colored("  No presets found!", RED)
            return

    prompt_file = preset.get("prompt_file", "prompt_basic.md")
    try:
        prompt = _load_prompt(prompt_file)
    except FileNotFoundError:
        prompt = BASIC_PROMPT

    max_iterations = preset.get("max_iterations", 50)
    compaction_cfg = preset.get("compaction", {})
    compaction_enabled = compaction_cfg.get("enabled", False)
    compact_threshold = compaction_cfg.get("threshold", 80)
    keep_recent = compaction_cfg.get("keep_recent", 16)

    engine = ChatEngine(
        llm_caller=chat_stream,
        tool_executor=execute_tool,
        is_dangerous=is_dangerous,
        get_permission=get_permission,
        conv_history=_conv_history,
        get_usage_pct=_get_usage_pct,
    )

    engine.run(
        goal=goal,
        prompt=prompt,
        tools=TOOLS,
        max_iterations=max_iterations,
        compaction=compaction_enabled,
        compact_threshold=compact_threshold,
        keep_recent=keep_recent,
    )


def conversational_turn(user_input: str) -> None:
    """Append user message to history and run the chat engine."""
    _conv_history.append({"role": "user", "content": user_input})
    run_chat(user_input)


# ---------------------------------------------------------------------------
# REPL helpers
# ---------------------------------------------------------------------------

def _get_terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return 80


def _get_terminal_height() -> int:
    try:
        return os.get_terminal_size().lines
    except Exception:
        return 24


def draw_status_bar() -> None:
    """Draw a persistent status bar at the bottom of the terminal."""
    settings = get_settings()
    usage = get_token_usage()
    width = _get_terminal_width()
    height = _get_terminal_height()

    pct = usage.percentage
    bar_width = 12
    filled = int(bar_width * pct / 100)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
    if pct < 50:
        bar_color = GREEN
    elif pct < 80:
        bar_color = YELLOW
    else:
        bar_color = RED

    ctx_str = f"{bar_color}{bar}{RESET} {pct:.0f}%"
    tokens_str = f"{usage.total_tokens:,}/{settings.context_size:,}"

    model_name = settings.model or "(no model)"
    if len(model_name) > 25:
        model_name = model_name[:22] + "..."
    preset_name = settings.preset or "?"
    turn_count = len([m for m in _conv_history if m.get("role") == "user"])

    inner = f" Stoat {DIM}│{RESET} {preset_name} {DIM}│{RESET} {model_name} {DIM}│{RESET} {ctx_str} ({tokens_str}) {DIM}│{RESET} turn {turn_count} "

    save = "\033[s"
    restore = "\033[u"
    move_bottom = f"\033[{height};1H"
    clear_line = "\033[2K"

    ansi_re = re.compile(r'\033\[[0-9;]*m')
    visible_len = len(ansi_re.sub('', inner))
    pad = max(0, width - visible_len)
    padded = inner + " " * pad

    sys.stdout.write(f"{save}{move_bottom}{clear_line}{DIM}{padded}{RESET}{restore}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

def handle_command(cmd: str) -> bool | None:
    """Handle a slash command. Returns True if handled, None to exit, False if unknown."""
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    settings = get_settings()

    if command in ('/help', '/h', '/?'):
        print_colored("\n  Available commands:", BOLD)
        print_colored("    /help, /h        Show this help", DIM)
        print_colored("    /ctx <size>      Set context window (e.g., /ctx 16384)", DIM)
        print_colored("    /temp <value>    Set temperature (e.g., /temp 0.5)", DIM)
        print_colored("    /model <name>    Load a model from models/ (restarts server)", DIM)
        print_colored("    /models          List local .gguf models", DIM)
        print_colored("    /status          Show current settings", DIM)
        print_colored("    /permissions     Show/set tool permissions", DIM)
        print_colored("    /presets         List available presets", DIM)
        print_colored("    /preset <name>   Switch preset", DIM)
        print_colored("    /backend <name>  Switch GPU backend (cuda, vulkan, cpu)", DIM)
        print_colored("    /compact [n]     Compact history (keep last n messages, default 6)", DIM)
        print_colored("    /clear           Clear conversation history", DIM)
        print_colored("    /exit, exit      Quit", DIM)
        print_colored("\n  Esc                Interrupt current generation", DIM)
        return True

    elif command == '/ctx':
        if not arg:
            print_colored(f"  Context: {settings.context_size}", CYAN)
            print_colored("  Usage: /ctx <size>  (e.g., /ctx 16384)", DIM)
            return True
        try:
            new_ctx = int(arg)
            if new_ctx < 1024:
                print_colored("  Warning: context size below 1024 is very small", YELLOW)
            elif new_ctx > 131072:
                print_colored("  Warning: context size very large, may cause OOM", YELLOW)
            settings.context_size = new_ctx
            print_colored(f"  Context set to {new_ctx}", GREEN)
        except ValueError:
            print_colored(f"  Invalid number: {arg}", RED)
        return True

    elif command == '/temp':
        if not arg:
            print_colored(f"  Temperature: {settings.temperature}", CYAN)
            print_colored("  Usage: /temp <value>  (0.0 - 2.0)", DIM)
            return True
        try:
            new_temp = float(arg)
            if not 0.0 <= new_temp <= 2.0:
                print_colored("  Warning: temperature typically between 0.0 and 2.0", YELLOW)
            settings.temperature = new_temp
            print_colored(f"  Temperature set to {new_temp}", GREEN)
        except ValueError:
            print_colored(f"  Invalid number: {arg}", RED)
        return True

    elif command == '/model':
        if not arg:
            print_colored(f"  Model: {settings.model}", CYAN)
            loaded = "running" if is_server_running() else "not loaded"
            print_colored(f"  Server: {loaded}", CYAN)
            print_colored("  Usage: /model <name>  (use /models to list)", DIM)
            return True
        local = list_local_models()
        matches = [m for m in local if arg.lower() in m.lower()]
        if len(matches) == 1:
            chosen = matches[0]
        elif arg in local:
            chosen = arg
        else:
            if matches:
                print_colored(f"  Multiple matches: {', '.join(matches)}", YELLOW)
            else:
                print_colored(f"  Model '{arg}' not found in models/", RED)
                print_colored(f"  Use /models to list available models", DIM)
            return True
        print_colored(f"  Loading {chosen}...", DIM)
        if load_model(chosen, ctx_size=settings.context_size):
            settings.model = chosen
            print_colored(f"  Model loaded: {chosen}", GREEN)
        else:
            print_colored(f"  Failed to load {chosen}", RED)
        return True

    elif command == '/models':
        local = list_local_models()
        if local:
            print_colored("\n  Local models (models/):", BOLD)
            for m in local:
                mpath = os.path.join(get_models_dir(), m)
                size_mb = os.path.getsize(mpath) / (1024 ** 2)
                meta = get_gguf_metadata(mpath)
                ctx_info = f", ctx {meta['context_length']//1024}K" if meta.get("context_length") else ""
                quant_info = f", {meta['quantization']}" if meta.get("quantization") else ""
                marker = f" {CYAN}<- loaded{RESET}" if m == settings.model and is_server_running() else ""
                print(f"    {m} ({size_mb:,.0f} MB{quant_info}{ctx_info}){marker}")
        else:
            print_colored("  No models in models/", DIM)
            print_colored(f"  Place .gguf files in: {get_models_dir()}/", DIM)
        return True

    elif command == '/status':
        print_colored("\n  Current settings:", BOLD)
        print_colored(f"    Model:       {settings.model}", CYAN)
        print_colored(f"    Context:     {settings.context_size}", CYAN)
        print_colored(f"    Temperature: {settings.temperature}", CYAN)
        print_colored(f"    Preset:      {settings.preset}", CYAN)
        print_colored(f"    Backend:     {get_backend().upper()}", CYAN)
        usage = get_token_usage()
        print_colored(f"    Tokens:      {usage.total_tokens:,}/{settings.context_size:,} ({usage.percentage:.0f}%)", CYAN)
        turns = len([m for m in _conv_history if m.get("role") == "user"])
        print_colored(f"    Turns:       {turns}", CYAN)
        return True

    elif command == '/compact':
        keep_recent = 6
        if arg:
            try:
                keep_recent = int(arg)
            except ValueError:
                pass

        if len(_conv_history) <= keep_recent + 1:
            print_colored(f"  Nothing to compact ({len(_conv_history)} messages, keeping {keep_recent})", DIM)
            return True

        cutoff = max(0, len(_conv_history) - keep_recent)
        old_msgs = _conv_history[:cutoff]
        recent_msgs = _conv_history[cutoff:]

        summary_parts = []
        files_mentioned = set()
        actions = []
        for msg in old_msgs:
            role = msg.get("role", "")
            content = msg.get("content") or ""

            if role == "user":
                short = content[:150].replace("\n", " ").strip()
                summary_parts.append(f"- User: {short}")
            elif role == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        name = func.get("name", "?")
                        tc_args = func.get("arguments", {})
                        if isinstance(tc_args, str):
                            try:
                                tc_args = json.loads(tc_args)
                            except Exception:
                                tc_args = {}
                        if name == "write_file":
                            path = tc_args.get("path", "?")
                            files_mentioned.add(path)
                            actions.append(f"Wrote {path}")
                        elif name == "edit_file":
                            path = tc_args.get("path", "?")
                            files_mentioned.add(path)
                            actions.append(f"Edited {path}")
                        elif name == "read_file":
                            path = tc_args.get("path", "?")
                            files_mentioned.add(path)
                            actions.append(f"Read {path}")
                        elif name == "shell":
                            actions.append(f"Ran: {tc_args.get('command', '?')[:50]}")
                        elif name == "list_dir":
                            actions.append("Listed directory")
                        else:
                            actions.append(f"Called {name}")
                elif content:
                    short = content[:100].replace("\n", " ").strip()
                    summary_parts.append(f"- Assistant: {short}")

        summary = "## Conversation summary (compacted)\n"
        if summary_parts:
            summary += "\n".join(summary_parts) + "\n"
        if actions:
            summary += "\nActions taken: " + "; ".join(actions) + "\n"
        if files_mentioned:
            summary += f"\nFiles touched: {', '.join(sorted(files_mentioned))}\n"

        new_history = [{"role": "assistant", "content": summary}] + recent_msgs
        _conv_history.clear()
        _conv_history.extend(new_history)

        before_count = len(old_msgs) + len(recent_msgs)
        after_count = len(new_history)

        print_colored(f"\n  [Compact] {before_count} messages -> {after_count} (kept last {keep_recent})", CYAN)
        if actions:
            print_colored("  Summary of compacted history:", DIM)
            for a in actions[:10]:
                print_colored(f"    -> {a}", DIM)
            if len(actions) > 10:
                print_colored(f"    ... and {len(actions) - 10} more", DIM)
        if files_mentioned:
            print_colored(f"  Files: {', '.join(sorted(files_mentioned))}", DIM)

        reset_token_usage()
        return True

    elif command == '/clear':
        reset_token_usage()
        reset_conv_history()
        print_colored("  Conversation cleared", GREEN)
        return True

    elif command == '/permissions':
        if not arg:
            perms = get_all_permissions()
            print_colored("\n  Tool permissions:", BOLD)
            for tool, value in sorted(perms.items()):
                if value is True:
                    label = "allowed"
                elif value is False:
                    label = "denied"
                else:
                    label = "ask"
                print_colored(f"    {tool + ':':16s} {label}", CYAN)
            return True

        perm_parts = arg.split()
        if len(perm_parts) != 2 or perm_parts[0] not in ALL_TOOL_NAMES:
            print_colored("  Usage: /permissions [tool] [y/n/ask]", DIM)
            print_colored(f"  Tools: {', '.join(sorted(ALL_TOOL_NAMES))}", DIM)
            return True

        tool_name, value_str = perm_parts
        if value_str in ('y', 'yes', 'allow'):
            set_permission(tool_name, True)
            print_colored(f"  {tool_name}: auto-allowed", GREEN)
        elif value_str in ('n', 'no', 'deny'):
            set_permission(tool_name, False)
            print_colored(f"  {tool_name}: denied", RED)
        elif value_str == 'ask':
            set_permission(tool_name, None)
            print_colored(f"  {tool_name}: ask each time (default)", YELLOW)
        else:
            print_colored("  Value must be y, n, or ask", RED)
        return True

    elif command == '/presets':
        presets = list_presets()
        if presets:
            print_colored("\n  Available presets:", BOLD)
            for p in presets:
                marker = " <-" if p == settings.preset else ""
                print_colored(f"    {p}{marker}", CYAN if marker else DIM)
        else:
            print_colored("  No presets found in .agent-stoat/presets/", DIM)
        return True

    elif command == '/preset':
        if not arg:
            print_colored(f"  Current preset: {settings.preset}", CYAN)
            print_colored("  Usage: /preset <name>  (use /presets to list)", DIM)
            return True
        presets = list_presets()
        if arg in presets:
            settings.preset = arg
            try:
                preset = load_preset(arg)
                print_colored(f"  Preset: {preset.get('name', arg)}", GREEN)
                print_colored(f"  {preset.get('description', '')}", DIM)
            except Exception:
                print_colored(f"  Switched to {arg}", GREEN)
        else:
            print_colored(f"  Preset '{arg}' not found. Use /presets to list.", RED)
        return True

    elif command == '/backend':
        current = get_backend()
        if not arg:
            print_colored(f"  Current backend: {current.upper()}", CYAN)
            print_colored("  Usage: /backend <cuda|vulkan|cpu>", DIM)
            return True
        arg_lower = arg.lower()
        if arg_lower == current:
            print_colored(f"  Already using {current.upper()}", DIM)
            return True
        if set_backend(arg_lower):
            print_colored(f"  Backend switched to {arg_lower.upper()}", GREEN)
            if is_server_running():
                print_colored("  Restarting server with new backend...", DIM)
                stop_server()
                load_model(settings.model, ctx_size=settings.context_size)
        else:
            print_colored(f"  Could not switch to '{arg}'. Binary not found.", RED)
            print_colored(f"  Place the {arg} build in: llama_cpp_backends/{arg}/", DIM)
        return True

    elif command in ('/exit', '/quit'):
        return None

    return False


# ---------------------------------------------------------------------------
# Startup menu
# ---------------------------------------------------------------------------

def startup_menu() -> None:
    """Interactive startup menu."""
    settings = get_settings()

    if not is_server_running():
        if not _load_model_interactive(MODEL_FILE):
            print_colored("  No model loaded. Use [2] to load one.", YELLOW)

    while True:
        server_up = is_server_running()
        model_tag = _shorten_model_name(settings.model) if server_up else "not loaded"
        model_color = GREEN if server_up else RED
        preset_label = settings.preset
        backend_label = get_backend().upper()

        rows = [
            ("[1] Start", None, ""),
            ("[2] Load Model       ", model_tag, model_color),
            ("[3] Set Context Window", None, ""),
            ("[4] Set Permissions", None, ""),
            ("[5] Switch Preset    ", preset_label, ""),
            ("[6] Switch Backend   ", backend_label, ""),
        ]

        def _visible_len(text, tag):
            return len(f"  {text}({tag})  " if tag else f"  {text}  ")
        W = max(_visible_len(t, tag) for t, tag, _ in rows)
        W = max(W, 20)

        def menu_line(text, tag=None, tag_color=""):
            if tag:
                visible = f"  {text}({tag})  "
                colored = f"  {text}({tag_color}{tag}{RESET})  "
            else:
                visible = f"  {text}  "
                colored = visible
            pad = W - len(visible)
            return f"  {BOLD}│{RESET}{colored}{' ' * max(0, pad)}{BOLD}│{RESET}"

        print(f"\n  {BOLD}╭{'─' * W}╮{RESET}")
        print(f"  {BOLD}│{RESET}{'Agent Stoat':^{W}}{BOLD}│{RESET}")
        print(f"  {BOLD}├{'─' * W}┤{RESET}")
        for text, tag, color in rows:
            print(menu_line(text, tag, color))
        print(f"  {BOLD}╰{'─' * W}╯{RESET}")

        choice = input("  Select [1-6]: ").strip()

        if choice in ('', '1'):
            if not server_up:
                print_colored("  No model loaded! Use [2] to load one first.", YELLOW)
                continue
            return

        elif choice == '2':
            _load_model_interactive()

        elif choice == '3':
            info = get_model_info()
            max_ctx = info.get("context_length")
            param_count = info.get("parameter_count")
            quant = info.get("quantization", "")

            print_colored(f"\n  Model:           {CYAN}{settings.model}{RESET}")
            print_colored(f"  Current context: {CYAN}{settings.context_size}{RESET}")
            if max_ctx:
                print_colored(f"  Model max:       {CYAN}{max_ctx:,}{RESET}")
            if param_count:
                print_colored(f"  Model params:    {CYAN}{param_count / 1e9:.1f}B{RESET}")
            if quant:
                print_colored(f"  Quantization:    {CYAN}{quant}{RESET}")

            vram = get_vram_info()
            if vram:
                model_loaded = settings.model in get_loaded_models()
                loaded_label = f"{GREEN}loaded{RESET}" if model_loaded else f"{DIM}not loaded{RESET}"
                print_colored(f"\n  GPU VRAM:        {CYAN}{vram['free_mb']:,} MB free / {vram['total_mb']:,} MB total{RESET}")
                print_colored(f"  Model in VRAM:   {loaded_label}")
                model_path = find_model(settings.model)
                rec = estimate_recommended_ctx(info, vram, model_loaded=model_loaded, model_file_path=model_path)
                if rec:
                    print_colored(f"  Model est. size: {CYAN}~{rec['model_vram_mb']:,.0f} MB{RESET}")
                    if 'kv_bytes_per_token' in rec:
                        accuracy = '(exact)' if rec.get('architecture_known') else '(estimated)'
                        print_colored(f"  KV cache/token:  {CYAN}{rec['kv_bytes_per_token']:,} bytes {accuracy}{RESET}")
                    if rec.get("recommended_ctx"):
                        vram_ctx = rec.get("vram_ctx", 0)
                        model_max = rec.get("model_max_ctx")
                        if model_max and vram_ctx > model_max:
                            print_colored(f"  Recommended ctx: {GREEN}{rec['recommended_ctx']:,} (MAX){RESET}")
                            print_colored(f"  Note: VRAM could support {vram_ctx:,}, but model max is {model_max:,}", DIM)
                        else:
                            print_colored(f"  Recommended ctx: {GREEN}{rec['recommended_ctx']:,}{RESET}")
                    else:
                        print_colored(f"  Recommended ctx: {RED}insufficient VRAM{RESET}")
                else:
                    rec = {}
            else:
                print_colored(f"\n  GPU VRAM:        {DIM}unavailable (nvidia-smi not found){RESET}")
                rec = {}

            overhead_tokens = (len(BASIC_PROMPT) + len(json.dumps(TOOLS))) // 4
            usable = settings.context_size - overhead_tokens
            print_colored(f"\n  Prompt overhead: ~{overhead_tokens} tokens (system prompt + tool defs)", DIM)
            print_colored(f"  Usable for chat: ~{max(0, usable)} tokens at current setting", DIM)

            ctx_input = input(f"\n  New context size (Enter to keep {settings.context_size}): ").strip()
            if ctx_input:
                try:
                    new_ctx = int(ctx_input)
                    if new_ctx < 1024:
                        print_colored("  Warning: context size below 1024 is very small", YELLOW)
                    if max_ctx and new_ctx > max_ctx:
                        print_colored(f"  Warning: exceeds model max of {max_ctx:,}", YELLOW)
                    if rec.get("recommended_ctx") and new_ctx > rec["recommended_ctx"]:
                        print_colored(f"  Warning: exceeds VRAM-based recommendation of {rec['recommended_ctx']:,}", YELLOW)
                    settings.context_size = new_ctx
                    print_colored(f"  Context set to {new_ctx}", GREEN)
                except ValueError:
                    print_colored("  Invalid number", RED)

        elif choice == '4':
            perm_cycle = [None, True, False]
            tool_list = sorted(ALL_TOOL_NAMES)

            while True:
                perms = get_all_permissions()
                print_colored("\n  Tool Permissions:", BOLD)
                for i, tool in enumerate(tool_list, 1):
                    value = perms.get(tool)
                    if value is True:
                        label = f"{GREEN}allowed{RESET}"
                    elif value is False:
                        label = f"{RED}denied{RESET}"
                    else:
                        default_note = " (default)" if tool in DANGEROUS_TOOLS else ""
                        label = f"{YELLOW}ask{RESET}{DIM}{default_note}{RESET}"
                    print(f"    [{i:2d}] {tool + ':':16s} {label}")

                perm_choice = input(f"  Toggle [1-{len(tool_list)}] (Enter to go back): ").strip()
                if not perm_choice:
                    break

                try:
                    idx = int(perm_choice) - 1
                    if 0 <= idx < len(tool_list):
                        tool = tool_list[idx]
                        current = perms.get(tool)
                        current_idx = perm_cycle.index(current)
                        new_value = perm_cycle[(current_idx + 1) % len(perm_cycle)]
                        set_permission(tool, new_value)
                    else:
                        print_colored("  Invalid selection", RED)
                except ValueError:
                    print_colored("  Invalid selection", RED)

        elif choice == '5':
            presets = list_presets()
            if not presets:
                print_colored("  No presets found", RED)
                continue
            print_colored("\n  Available presets:", BOLD)
            for i, p in enumerate(presets, 1):
                marker = f"  {CYAN}<- current{RESET}" if p == settings.preset else ""
                print(f"    [{i}] {p}{marker}")
            pchoice = input(f"\n  Select [1-{len(presets)}] (Enter to cancel): ").strip()
            if pchoice:
                try:
                    idx = int(pchoice) - 1
                    if 0 <= idx < len(presets):
                        settings.preset = presets[idx]
                        print_colored(f"  Preset: {settings.preset}", GREEN)
                    else:
                        print_colored("  Invalid selection", RED)
                except ValueError:
                    print_colored("  Invalid selection", RED)

        elif choice == '6':
            backends = ["cuda", "vulkan", "cpu"]
            current = get_backend()
            print_colored("\n  Available backends:", BOLD)
            for i, b in enumerate(backends, 1):
                marker = f"  {CYAN}<- current{RESET}" if b == current else ""
                print(f"    [{i}] {b.upper()}{marker}")
            bchoice = input(f"\n  Select [1-{len(backends)}] (Enter to cancel): ").strip()
            if bchoice:
                try:
                    idx = int(bchoice) - 1
                    if 0 <= idx < len(backends):
                        chosen = backends[idx]
                        if chosen == current:
                            print_colored(f"  Already using {chosen.upper()}", DIM)
                        elif set_backend(chosen):
                            print_colored(f"  Backend: {chosen.upper()}", GREEN)
                            if server_up:
                                print_colored("  Restarting server with new backend...", DIM)
                                stop_server()
                                load_model(settings.model, ctx_size=settings.context_size)
                        else:
                            print_colored(f"  No {chosen.upper()} binary found.", RED)
                            print_colored(f"  Place the {chosen} build in: llama_cpp_backends/{chosen}/", DIM)
                    else:
                        print_colored("  Invalid selection", RED)
                except ValueError:
                    print_colored("  Invalid selection", RED)

        else:
            print_colored("  Invalid selection", RED)


# ---------------------------------------------------------------------------
# Startup / header
# ---------------------------------------------------------------------------

def print_header():
    settings = get_settings()
    from llm_server import SERVER_HOST
    model_display = settings.model if is_server_running() else "(none loaded)"
    print_colored(f"\n{BOLD}╭{'─' * 48}╮{RESET}")
    print_colored(f"{BOLD}│  Agent Stoat{' ' * 35}│{RESET}")
    print_colored(f"{BOLD}╰{'─' * 48}╯{RESET}")
    backend = get_backend().upper()
    print_colored(f"  Backend: {backend}", DIM)
    print_colored(f"  Model:   {model_display}", DIM)
    print_colored(f"  Context: {settings.context_size}", DIM)
    print_colored(f"  Server:  {SERVER_HOST}", DIM)
    print_colored(f"  Models:  {get_models_dir()}", DIM)
    print_colored(f"  Workdir: {os.getcwd()}", DIM)
    print_colored(f"  Preset:  {settings.preset}", DIM)
    print()


def _load_permissions_overlay() -> None:
    """Apply tool permission overrides from .agent-stoat/settings.json."""
    settings_path = os.path.join(AGENT_DATA_DIR, "settings.json")
    if not os.path.exists(settings_path):
        return
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for tool_name, value_str in data.get("permissions", {}).items():
            if tool_name not in ALL_TOOL_NAMES:
                continue
            if value_str == "allow":
                set_permission(tool_name, True)
            elif value_str == "deny":
                set_permission(tool_name, False)
            else:
                set_permission(tool_name, None)
    except Exception:
        pass


def _select_working_dir() -> str:
    """Let user choose between a new working directory or an existing one."""
    from datetime import datetime
    os.makedirs(WORKING_DIR, exist_ok=True)

    # List existing session directories
    existing = sorted(
        [d for d in os.listdir(WORKING_DIR)
         if os.path.isdir(os.path.join(WORKING_DIR, d))],
        reverse=True,  # Most recent first
    )

    if existing:
        print_colored("\n  Working Directory:", BOLD)
        print_colored("    [1] Create new session folder", DIM)
        for i, d in enumerate(existing[:9], 2):
            print(f"    [{i}] {d}")
        if len(existing) > 9:
            print_colored(f"    ... and {len(existing) - 9} more", DIM)

        choice = input(f"  Select [1-{min(len(existing) + 1, 10)}] (Enter for 1): ").strip()
        if choice and choice != '1':
            try:
                idx = int(choice) - 2
                if 0 <= idx < len(existing):
                    chosen = os.path.join(WORKING_DIR, existing[idx])
                    print_colored(f"  Resuming: {existing[idx]}", GREEN)
                    return chosen
            except ValueError:
                pass

    timestamp = datetime.now().strftime("%Y-%m-%d %I-%M-%S %p")
    session_dir = os.path.join(WORKING_DIR, f"Stoat Working Dir - {timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def _load_model_interactive(model_file: str = None) -> bool:
    """Prompt the user to select and load a model. Returns True on success."""
    settings = get_settings()
    local_models = list_local_models()

    if model_file and find_model(model_file):
        settings.model = model_file
        return load_model(model_file, ctx_size=settings.context_size)

    if not local_models:
        print_colored(f"\n  No models found in {get_models_dir()}/", YELLOW)
        print_colored(f"  Default: {MODEL_FILE}", DIM)
        return load_model(
            MODEL_FILE,
            ctx_size=settings.context_size,
            model_repo=MODEL_REPO,
        )

    if len(local_models) == 1:
        chosen = local_models[0]
        settings.model = chosen
        return load_model(chosen, ctx_size=settings.context_size)

    print_colored("\n  Available models:", BOLD)
    for i, m in enumerate(local_models, 1):
        marker = f"  {CYAN}<- configured{RESET}" if m == model_file else ""
        mpath = os.path.join(get_models_dir(), m)
        size_mb = os.path.getsize(mpath) / (1024 ** 2)
        meta = get_gguf_metadata(mpath)
        ctx_info = f", ctx {meta['context_length']//1024}K" if meta.get("context_length") else ""
        quant_info = f", {meta['quantization']}" if meta.get("quantization") else ""
        print(f"    [{i}] {m} ({size_mb:,.0f} MB{quant_info}{ctx_info}){marker}")

    choice = input(f"\n  Select [1-{len(local_models)}] (Enter for 1): ").strip()
    if not choice:
        idx = 0
    else:
        try:
            idx = int(choice) - 1
        except ValueError:
            idx = 0

    if 0 <= idx < len(local_models):
        chosen = local_models[idx]
        settings.model = chosen
        return load_model(chosen, ctx_size=settings.context_size)

    print_colored("  Invalid selection", RED)
    return False


def main():
    """Main REPL entry point."""
    _load_permissions_overlay()

    if not setup_environment():
        print_colored("\n  Server binary not found. Cannot continue.", RED)
        print_colored("    See setup instructions above.", RED)
        return

    session_dir = _select_working_dir()
    os.chdir(session_dir)

    print_header()
    startup_menu()

    print()
    print_colored("  Type /help for a list of all slash commands.", DIM)
    print_colored("  Press Esc to interrupt generation mid-stream.", DIM)
    print_colored("─" * 50, DIM)

    _history_file = os.path.join(AGENT_DATA_DIR, "history")
    if readline:
        try:
            readline.read_history_file(_history_file)
        except FileNotFoundError:
            pass
        readline.set_history_length(1000)

    try:
        while True:
            try:
                draw_status_bar()
                user_input = input(f"\n{BOLD}You >{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl+C or EOF — exit cleanly
                break

            if not user_input:
                continue

            if user_input.lower() in ('exit', 'quit'):
                break

            if user_input.startswith('/'):
                result = handle_command(user_input)
                if result is None:
                    break
                if result:
                    draw_status_bar()
                    continue

            try:
                conversational_turn(user_input)
            except KeyboardInterrupt:
                # Ctrl+C during generation — abort this turn, stay in REPL
                print(f"\n{YELLOW}  (Interrupted){RESET}")
            draw_status_bar()

    finally:
        print_colored("\n  Stopping server...", DIM)
        stop_server()
        if readline:
            try:
                readline.write_history_file(_history_file)
            except Exception:
                pass

    print_colored("  Goodbye!", DIM)


if __name__ == "__main__":
    main()
