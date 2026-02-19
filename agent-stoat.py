#!/usr/bin/env python3
"""Stoat — local coding agent with tool calling."""

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

from config import OLLAMA_HOST, SYSTEM_PROMPT, COMPACT_THRESHOLD, COMPACT_EMERGENCY
from ollama_client import (
    chat, chat_stream, check_connection, get_token_usage,
    get_settings, reset_token_usage, list_models, get_model_info,
    get_vram_info, estimate_recommended_ctx, get_loaded_models
)
from tool_parser import extract_tool_calls, get_response_content
from tools import (
    TOOLS, execute_tool, is_dangerous, get_permission,
    set_permission, get_all_permissions, DANGEROUS_TOOLS, SCRATCHPAD_PATH, CONTEXT_FILE_PATH,
    AGENT_DATA_DIR, WORKING_DIR
)

# ANSI colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def compact_history(history: list[dict]) -> list[dict]:
    """Distill conversation history into context.md and return a compacted history."""
    if len(history) <= 2:
        return history

    system = history[0]

    # Format history with generous per-message limit
    history_text = ""
    for msg in history[1:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:2000]
        if content:
            history_text += f"[{role}]:\n{content}\n\n"

    summary_prompt = [
        {"role": "system", "content": "You are a precise technical assistant. Your job is to distill conversation history into a structured state file."},
        {"role": "user", "content": (
            "Distill the following conversation into a comprehensive context file.\n"
            "Be thorough — this file is the agent's only memory after compaction.\n\n"
            "Structure your output as markdown with these sections:\n"
            "## Current Goal\n"
            "What the user asked for and what the agent is working toward.\n\n"
            "## Files Created / Modified\n"
            "Every file touched: its path, purpose, and a meaningful summary of its content.\n\n"
            "## Key Findings & Decisions\n"
            "Important discoveries, errors encountered and how they were resolved, design choices made.\n\n"
            "## Current State\n"
            "What was most recently completed. Where execution stopped.\n\n"
            "## Next Steps\n"
            "The immediate next action(s) to continue the task.\n\n"
            f"---\n\n{history_text}"
        )}
    ]

    print_colored("  Compacting context...", DIM)
    response = chat_stream(summary_prompt, tools=None, print_output=False)
    distilled = response["message"]["content"]

    # Write distillation to context.md
    try:
        with open(CONTEXT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(distilled)
    except Exception as e:
        print_colored(f"  ⚠ Could not write context file: {e}", YELLOW)

    return [system]


def _read_file_silent(path: str) -> str:
    """Read a file, returning empty string if missing."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ""


def read_scratchpad_content() -> str:
    """Read scratchpad file directly (internal use, bypasses tool system)."""
    return _read_file_silent(SCRATCHPAD_PATH)


def check_and_compact(history: list[dict]) -> list[dict]:
    """Auto-compact history if context usage exceeds threshold, re-injecting persistent state."""
    usage = get_token_usage()
    pct = usage.percentage

    if pct < COMPACT_THRESHOLD:
        return history

    label = "emergency" if pct >= COMPACT_EMERGENCY else "auto"
    print_colored(f"\n  ⚡ Context at {pct:.0f}% — {label}-compacting...", YELLOW)

    history = compact_history(history)

    # Re-inject both persistent state files
    context = _read_file_silent(CONTEXT_FILE_PATH)
    scratchpad = _read_file_silent(SCRATCHPAD_PATH)

    parts = [f"[Context compacted at {pct:.0f}% usage — reloading persistent state]"]

    if context:
        parts.append(f"## Distilled Context\n\n{context}")
    if scratchpad:
        parts.append(f"## Agent Scratchpad\n\n{scratchpad}")
    if not context and not scratchpad:
        parts.append("No saved state found. Wait for the next user instruction.")

    parts.append("Continue from where you left off.")
    history.append({"role": "user", "content": "\n\n".join(parts)})

    flags = []
    if context:
        flags.append("context.md")
    if scratchpad:
        flags.append("scratchpad")
    print_colored(f"  ✓ Compacted — re-injected: {', '.join(flags) if flags else 'nothing'}", GREEN)
    return history


def compact_display(text: str, max_len: int = 300) -> str:
    """Collapse whitespace and truncate for compact terminal display."""
    # Collapse all runs of whitespace (newlines, spaces, tabs) into single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def print_colored(text: str, color: str = RESET) -> None:
    """Print text with ANSI color."""
    print(f"{color}{text}{RESET}")


def print_context_status() -> None:
    """Print the current context window usage."""
    usage = get_token_usage()
    print(f"\n{DIM}Context: {usage.format_status()}{RESET}")


def handle_command(cmd: str, history: list[dict]) -> tuple[bool, list[dict]]:
    """
    Handle slash commands.

    Returns:
        (handled, history) - handled is True if command was processed
    """
    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None

    settings = get_settings()

    if command in ('/help', '/h', '/?'):
        print_colored("\n  Available commands:", BOLD)
        print_colored("    /help, /h        Show this help", DIM)
        print_colored("    /ctx <size>      Set context window (e.g., /ctx 16384)", DIM)
        print_colored("    /temp <value>    Set temperature (e.g., /temp 0.5)", DIM)
        print_colored("    /model <name>    Switch model (e.g., /model qwen2.5:7b)", DIM)
        print_colored("    /models          List available models", DIM)
        print_colored("    /status          Show current settings", DIM)
        print_colored("    /compact         Summarize history to free context", DIM)
        print_colored("    /scratchpad      View agent scratchpad contents", DIM)
        print_colored("    /context         View compacted context memory", DIM)
        print_colored("    /iterations <n>  Set max steps per message (0 = unlimited)", DIM)
        print_colored("    /permissions     Show/set tool permissions", DIM)
        print_colored("    /clear           Clear conversation history", DIM)
        print_colored("    /exit, exit      Quit", DIM)
        print_colored("\n  Esc                Interrupt current generation", DIM)
        return True, history

    elif command == '/ctx':
        if not arg:
            print_colored(f"  Context: {settings.context_size}", CYAN)
            print_colored("  Usage: /ctx <size>  (e.g., /ctx 16384)", DIM)
            return True, history
        try:
            new_ctx = int(arg)
            if new_ctx < 1024:
                print_colored("  ⚠ Context size should be at least 1024", YELLOW)
            elif new_ctx > 131072:
                print_colored("  ⚠ Context size very large, may cause OOM", YELLOW)
            settings.context_size = new_ctx
            print_colored(f"  ✓ Context set to {new_ctx}", GREEN)
        except ValueError:
            print_colored(f"  ✗ Invalid number: {arg}", RED)
        return True, history

    elif command == '/temp':
        if not arg:
            print_colored(f"  Temperature: {settings.temperature}", CYAN)
            print_colored("  Usage: /temp <value>  (0.0 - 2.0)", DIM)
            return True, history
        try:
            new_temp = float(arg)
            if not 0.0 <= new_temp <= 2.0:
                print_colored("  ⚠ Temperature typically between 0.0 and 2.0", YELLOW)
            settings.temperature = new_temp
            print_colored(f"  ✓ Temperature set to {new_temp}", GREEN)
        except ValueError:
            print_colored(f"  ✗ Invalid number: {arg}", RED)
        return True, history

    elif command == '/model':
        if not arg:
            print_colored(f"  Model: {settings.model}", CYAN)
            print_colored("  Usage: /model <name>  (use /models to list)", DIM)
            return True, history
        settings.model = arg
        print_colored(f"  ✓ Model set to {arg}", GREEN)
        print_colored("  Note: /clear recommended after switching models", DIM)
        return True, history

    elif command == '/models':
        print_colored("\n  Fetching models...", DIM)
        models = list_models()
        if models:
            print_colored("  Available models:", BOLD)
            for m in models:
                marker = " ←" if m == settings.model else ""
                print_colored(f"    {m}{marker}", CYAN if marker else DIM)
        else:
            print_colored("  Could not fetch models", RED)
        return True, history

    elif command == '/status':
        iterations_display = "unlimited" if settings.max_iterations is None else str(settings.max_iterations)
        print_colored("\n  Current settings:", BOLD)
        print_colored(f"    Model:       {settings.model}", CYAN)
        print_colored(f"    Context:     {settings.context_size}", CYAN)
        print_colored(f"    Temperature: {settings.temperature}", CYAN)
        print_colored(f"    Iterations:  {iterations_display}", CYAN)
        print_context_status()
        return True, history

    elif command == '/compact':
        old_len = len(history)
        history = compact_history(history)
        new_len = len(history)
        print_colored(f"  ✓ Compacted {old_len} messages → {new_len}", GREEN)
        print_context_status()
        return True, history

    elif command == '/scratchpad':
        content = read_scratchpad_content()
        if content:
            print_colored("\n  Scratchpad contents:", BOLD)
            print(content)
        else:
            print_colored("  (Scratchpad is empty)", DIM)
        return True, history

    elif command == '/context':
        content = _read_file_silent(CONTEXT_FILE_PATH)
        if content:
            print_colored("\n  Compacted context memory:", BOLD)
            print(content)
        else:
            print_colored("  (No compaction has occurred yet)", DIM)
        return True, history

    elif command == '/iterations':
        if not arg:
            current = settings.max_iterations
            display = "unlimited" if current is None else str(current)
            print_colored(f"  Max iterations: {display}", CYAN)
            print_colored("  Usage: /iterations <n>  (0 = unlimited)", DIM)
            return True, history
        try:
            n = int(arg)
            if n < 0:
                print_colored("  ✗ Must be 0 or greater", RED)
            elif n == 0:
                settings.max_iterations = None
                print_colored("  ✓ Max iterations: unlimited", GREEN)
            else:
                settings.max_iterations = n
                print_colored(f"  ✓ Max iterations: {n}", GREEN)
        except ValueError:
            print_colored(f"  ✗ Invalid number: {arg}", RED)
        return True, history

    elif command == '/clear':
        history = [{"role": "system", "content": SYSTEM_PROMPT}]
        reset_token_usage()
        print_colored("  ✓ History cleared", GREEN)
        return True, history

    elif command == '/permissions':
        if not arg:
            # Display current permissions
            perms = get_all_permissions()
            print_colored("\n  Tool permissions:", BOLD)
            for tool, value in sorted(perms.items()):
                if value is True:
                    label = "allowed"
                elif value is False:
                    label = "denied"
                else:
                    label = "ask  (default)"
                print_colored(f"    {tool + ':':14s} {label}", CYAN)
            return True, history

        # Parse: /permissions <tool> <y/n/ask>
        perm_parts = arg.split()
        if len(perm_parts) != 2 or perm_parts[0] not in DANGEROUS_TOOLS:
            print_colored("  Usage: /permissions [tool] [y/n/ask]", DIM)
            print_colored(f"  Tools: {', '.join(sorted(DANGEROUS_TOOLS))}", DIM)
            return True, history

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
        return True, history

    elif command in ('/exit', '/quit'):
        return True, None  # Signal to exit

    return False, history


def agent_loop(user_input: str, history: list[dict]) -> list[dict]:
    """
    Run the agent loop for a single user input.

    Args:
        user_input: The user's message
        history: Conversation history (modified in place)

    Returns:
        Updated conversation history
    """
    # Add user message to history
    history.append({"role": "user", "content": user_input})

    while True:
        limit = get_settings().max_iterations
        iteration = 0

        while limit is None or iteration < limit:
            iteration += 1

            # Auto-compact if context is filling up
            history = check_and_compact(history)

            # Call Ollama
            response = chat(history, tools=TOOLS)

            # Check if user interrupted
            if response.get("interrupted"):
                content = get_response_content(response)
                if content:
                    history.append({
                        "role": "assistant",
                        "content": content
                    })
                print_colored("\n  ⚡ Interrupted", YELLOW)
                print_context_status()
                return history

            # Check for tool calls
            tool_calls = extract_tool_calls(response)

            # Get any text content from the response
            content = get_response_content(response)

            if not tool_calls:
                # No tools called - content already streamed, just update history
                history.append({
                    "role": "assistant",
                    "content": content
                })

                # Show context usage after response
                print_context_status()
                return history

            # Content already streamed, no need to print again

            # Add assistant message with tool calls to history
            history.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    }
                    for tc in tool_calls
                ]
            })

            # Execute each tool call
            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]

                print_colored(f"\n  ╭─ {name}", YELLOW)
                # Format args nicely
                for key, value in args.items():
                    display_value = str(value)
                    if len(display_value) > 60:
                        display_value = display_value[:60] + "..."
                    print_colored(f"  │  {key}: {display_value}", YELLOW)

                # Check permissions for dangerous tools
                if is_dangerous(name):
                    perm = get_permission(name)
                    if perm is None:  # Ask user
                        answer = input(f"  Allow {name}? [y/n]: ").strip().lower()
                        if answer != 'y':
                            result = "(Denied by user)"
                            print_colored(f"  ╰─ {result}", RED)
                            history.append({"role": "tool", "content": result})
                            continue
                    elif perm is False:  # Always denied
                        result = "(Blocked by permission settings)"
                        print_colored(f"  ╰─ {result}", RED)
                        history.append({"role": "tool", "content": result})
                        continue
                    # perm is True: auto-approved, fall through

                # Execute the tool
                result = execute_tool(name, args)

                # Show compact result preview
                print_colored(f"  ╰─ {compact_display(result)}", GREEN)

                # Add tool result to history
                history.append({
                    "role": "tool",
                    "content": result
                })

        # Limit reached — ask user whether to continue
        print_colored(f"\n  Reached {limit} steps.", YELLOW)
        try:
            answer = input("  Continue? [y/n]: ").strip().lower()
        except EOFError:
            answer = 'n'
        if answer != 'y':
            print_context_status()
            return history
        # else: outer loop repeats, resetting iteration counter


def startup_menu() -> None:
    """Interactive startup menu for configuring model and permissions."""
    settings = get_settings()

    while True:
        print_colored(f"\n  {BOLD}╭── Startup ──────────────────────────╮{RESET}")
        print_colored(f"  {BOLD}│{RESET}  [1] Start                          {BOLD}│{RESET}")
        print_colored(f"  {BOLD}│{RESET}  [2] Set Context Window             {BOLD}│{RESET}")
        print_colored(f"  {BOLD}│{RESET}  [3] Set Model                      {BOLD}│{RESET}")
        print_colored(f"  {BOLD}│{RESET}  [4] Set Permissions                {BOLD}│{RESET}")
        print_colored(f"  {BOLD}│{RESET}  [5] Set Max Iterations             {BOLD}│{RESET}")
        print_colored(f"  {BOLD}╰─────────────────────────────────────╯{RESET}")

        choice = input("  Select [1-5]: ").strip()

        if choice in ('', '1'):
            return

        elif choice == '2':
            # Context window submenu
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

            # VRAM-based recommendation
            vram = get_vram_info()
            if vram:
                model_loaded = settings.model in get_loaded_models()
                loaded_label = f"{GREEN}loaded{RESET}" if model_loaded else f"{DIM}not loaded{RESET}"
                print_colored(f"\n  GPU VRAM:        {CYAN}{vram['free_mb']:,} MB free / {vram['total_mb']:,} MB total{RESET}")
                print_colored(f"  Model in VRAM:   {loaded_label}")
                rec = estimate_recommended_ctx(info, vram, model_loaded=model_loaded)
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

            # Prompt overhead estimate
            overhead_tokens = (len(SYSTEM_PROMPT) + len(json.dumps(TOOLS))) // 4
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

        elif choice == '3':
            # Model selection submenu
            print_colored("\n  Fetching models...", DIM)
            models = list_models()
            if not models:
                print_colored("  Could not fetch models from Ollama", RED)
                continue

            print_colored("  Available models:", BOLD)
            for i, m in enumerate(models, 1):
                marker = f"  {CYAN}<- current{RESET}" if m == settings.model else ""
                print_colored(f"    [{i}] {m}{marker}", RESET)

            model_choice = input(f"  Select [1-{len(models)}] (Enter to cancel): ").strip()
            if model_choice:
                try:
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(models):
                        settings.model = models[idx]
                        print_colored(f"  Model set to {settings.model}", GREEN)
                    else:
                        print_colored("  Invalid selection", RED)
                except ValueError:
                    print_colored("  Invalid selection", RED)

        elif choice == '4':
            # Permissions submenu
            perm_cycle = [None, True, False]  # ask -> allowed -> denied -> ask
            tool_list = sorted(DANGEROUS_TOOLS)

            while True:
                perms = get_all_permissions()
                print_colored("\n  Tool Permissions:", BOLD)
                for i, tool in enumerate(tool_list, 1):
                    value = perms[tool]
                    if value is True:
                        label = f"{GREEN}allowed{RESET}"
                    elif value is False:
                        label = f"{RED}denied{RESET}"
                    else:
                        label = f"{YELLOW}ask{RESET} {DIM}(default){RESET}"
                    print(f"    [{i}] {tool + ':':14s} {label}")

                perm_choice = input(f"  Toggle [1-{len(tool_list)}] (Enter to go back): ").strip()
                if not perm_choice:
                    break

                try:
                    idx = int(perm_choice) - 1
                    if 0 <= idx < len(tool_list):
                        tool = tool_list[idx]
                        current = perms[tool]
                        current_idx = perm_cycle.index(current)
                        new_value = perm_cycle[(current_idx + 1) % len(perm_cycle)]
                        set_permission(tool, new_value)
                    else:
                        print_colored("  Invalid selection", RED)
                except ValueError:
                    print_colored("  Invalid selection", RED)

        elif choice == '5':
            current = settings.max_iterations
            display = "unlimited" if current is None else str(current)
            print_colored(f"\n  Current max iterations: {CYAN}{display}{RESET}")
            print_colored("  Enter a number, or 0 for unlimited.", DIM)
            val = input(f"  New value (Enter to keep {display}): ").strip()
            if val:
                try:
                    n = int(val)
                    if n < 0:
                        print_colored("  Must be 0 or greater", RED)
                    elif n == 0:
                        settings.max_iterations = None
                        print_colored("  Max iterations: unlimited", GREEN)
                    else:
                        settings.max_iterations = n
                        print_colored(f"  Max iterations: {n}", GREEN)
                except ValueError:
                    print_colored("  Invalid number", RED)

        else:
            print_colored("  Invalid selection", RED)


def print_header():
    """Print the startup header."""
    settings = get_settings()
    print_colored(f"\n{BOLD}╭{'─' * 48}╮{RESET}")
    print_colored(f"{BOLD}│  Agent Stoat{' ' * 35}│{RESET}")
    print_colored(f"{BOLD}╰{'─' * 48}╯{RESET}")
    print_colored(f"  Model:   {settings.model}", DIM)
    print_colored(f"  Context: {settings.context_size}", DIM)
    print_colored(f"  Ollama:  {OLLAMA_HOST}", DIM)
    print_colored(f"  Workdir: {WORKING_DIR}", DIM)
    print()


def main():
    """Main REPL entry point."""
    os.chdir(WORKING_DIR)
    print_header()

    # Check Ollama connection
    if not check_connection():
        print_colored(f"  ⚠ Cannot connect to Ollama at {OLLAMA_HOST}", RED)
        print_colored("    Make sure Ollama is running and accessible.", RED)
        print()
    else:
        print_colored("  ✓ Connected to Ollama", GREEN)
        print()

    startup_menu()

    print_context_status()
    print_colored("  Type /help for commands, Esc to interrupt", DIM)
    print_colored("─" * 50, DIM)

    # Conversation history with system prompt
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Configure readline history
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
                user_input = input(f"\n{BOLD}You ▸{RESET} ").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue

            if user_input.lower() in ('exit', 'quit'):
                break

            # Check for slash commands
            if user_input.startswith('/'):
                handled, history = handle_command(user_input, history)
                if history is None:  # Exit signal
                    break
                if handled:
                    continue

            # Run agent loop
            history = agent_loop(user_input, history)

    except KeyboardInterrupt:
        print("\n  Interrupted.")

    finally:
        # Save readline history
        if readline:
            try:
                readline.write_history_file(_history_file)
            except Exception:
                pass

    print_colored("\n  Goodbye!", DIM)


if __name__ == "__main__":
    main()
