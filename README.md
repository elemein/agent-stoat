# Agent Stoat

![Stoat](agent-stoat_scripts/agent-stoat.png)

Local-first coding agent built with small models in mind and tight VRAM budgets.

## Features

- **Streaming REPL** with colored output and command history
- **Tool calling** with robust multi-format parsing (native Ollama, XML tags, markdown JSON, raw JSON)
- **Core tools**: file read/write/edit, shell execution, web fetch, web search
- **Persistent scratchpad** — the agent maintains its own working notes across context compactions
- **Smart context compaction** — automatically distills history into a structured memory file when context fills up, then re-injects it so the agent continues seamlessly
- **Isolated working directory** — the agent works inside `agent-stoat_working-dir/`, separate from the agent code
- **Esc to interrupt** generation mid-stream
- **Startup menu** to configure model, context size, permissions, and iteration limit before entering the REPL
- **Runtime configuration** — switch models, adjust context size, temperature, and iteration limit on the fly

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- `requests` package (`pip install requests`)
- Windows, Linux, or WSL

## Setup

1. Install and start Ollama:
   ```bash
   ollama serve
   ```

2. Pull a model:
   ```bash
   ollama pull qwen2.5-coder:14b-instruct-q5_K_M
   ```

3. Install dependencies:
   ```bash
   pip install requests
   ```

4. Run the agent:
   ```bash
   python agent-stoat.py
   ```

On first run, two directories are created automatically:
- `agent-stoat/.agent-stoat/` — agent state (scratchpad, context memory, REPL history)
- `agent-stoat_working-dir/` — where the agent does all its work

## Startup Menu

On launch, an interactive menu lets you configure the agent before entering the REPL:

- **Start** (default) — press Enter or `1` to go straight into the REPL
- **Set Context Window** — view/change context size with estimates of token overhead and usable conversation space; queries `nvidia-smi` and recommends a context size based on free VRAM, model size, and KV cache math
- **Set Model** — pick from available Ollama models
- **Set Permissions** — toggle `write_file`, `edit_file`, and `shell` between `ask` (prompt each time), `allowed` (auto-approve), and `denied` (block)
- **Set Max Iterations** — set the maximum number of tool-calling steps per message (`0` = unlimited); when the limit is reached the agent prompts "Continue? [y/n]" rather than stopping silently

## Commands

| Command | Description |
|---|---|
| `/help, /h, /?` | Show available commands |
| `/model <name>` | Switch model |
| `/models` | List available Ollama models |
| `/ctx <size>` | Set context window size |
| `/temp <value>` | Set temperature (0.0 - 2.0) |
| `/iterations <n>` | Set max tool-calling steps per message (`0` = unlimited) |
| `/status` | Show current settings and context usage |
| `/compact` | Manually distill history to free context |
| `/scratchpad` | View the agent's scratchpad |
| `/context` | View the compacted context memory file |
| `/clear` | Clear conversation history |
| `/permissions` | Show/set tool permissions (e.g., `/permissions shell y`) |
| `/exit, /quit` | Quit |
| `Esc` | Interrupt current generation |

## Tools

| Tool | Description | Output Limit |
|---|---|---|
| `read_file` | Read file contents | 5000 chars per call; use `start_line`/`end_line` for large files |
| `write_file` | Create or overwrite a file | — |
| `edit_file` | Replace a unique string in a file (for surgical edits) | — |
| `list_dir` | List files and folders in a directory with sizes | 50 entries |
| `find_files` | Find files by name pattern recursively (e.g., `*.py`) | 30 results |
| `search_files` | Search for text/regex inside files with line numbers | 30 matches, 150 chars/line |
| `shell` | Execute a shell command and return output | 3000 chars |
| `web_fetch` | Fetch a URL and return clean text (HTML stripped) | 3000 chars |
| `web_search` | Search the web via DuckDuckGo | 5 results, 2000 chars total |
| `read_scratchpad` | Read the agent's persistent scratchpad | — |
| `update_scratchpad` | Overwrite the scratchpad with new state | — |

Tool outputs are truncated to keep context usage low for small local models. These limits are defined at the top of each function in `tools.py`.

## Context Management

The agent tracks token usage as a percentage of the context window. When usage crosses **70%**, context is automatically compacted:

1. The full conversation history is distilled into a structured markdown file (`.agent-stoat/context.md`) covering current goal, files modified, key decisions, and next steps
2. The scratchpad (`.agent-stoat/scratchpad.md`) is preserved as-is
3. Both are re-injected into the fresh history so the agent continues without losing state

The agent is also prompted to update its scratchpad after each step, providing a second layer of state that survives compaction.

Thresholds are configurable in `config.py`:
- `COMPACT_THRESHOLD = 70` — trigger auto-compaction (%)
- `COMPACT_EMERGENCY = 85` — force compaction if threshold was missed (%)

## Project Structure

```
agent-stoat/
  agent-stoat.py                  # Entry point — run this
  agent-stoat_scripts/
    config.py               # Model, host, and threshold configuration
    ollama_client.py        # Ollama API client with streaming and token tracking
    prompt.md               # System prompt — edit to change agent behaviour
    tool_parser.py          # Multi-format tool call extraction from LLM responses
    tools.py                # Tool definitions, implementations, and path constants
  agent-stoat_working-dir/  # Agent's isolated working directory
  .agent-stoat/             # Runtime state (scratchpad.md, context.md, history)
```

## Configuration

Edit `config.py` to change defaults:

- `SYSTEM_PROMPT` — loaded from `agent-stoat_scripts/prompt.md`; edit that file to change agent behaviour
- `MODEL` — Default model name
- `TEMPERATURE` — Generation temperature (default `0.7`)
- `NUM_CTX` — Context window size (default `8192`)
- `MAX_ITERATIONS` — Max tool-calling steps per message (default `20`)
- `COMPACT_THRESHOLD` — Auto-compact at this % of context (default `70`)
- `COMPACT_EMERGENCY` — Force compact at this % (default `85`)
- `OLLAMA_HOST` — Auto-detected (localhost on Windows/Linux, WSL gateway in WSL)
