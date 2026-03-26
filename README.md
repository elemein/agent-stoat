# Agent Stoat

![Stoat](agent-stoat_scripts/agent-stoat.png)

Local-first AI agent built with small models in mind. Powered by llama.cpp. No cloud, no subscriptions — runs entirely on your hardware.

## Two Modes

### 1. Interactive (Basic)
A capable, tool-using chat assistant. You ask, it acts. Reads and writes files, runs shell commands, searches the web, edits code. Each session has its own conversation history. No background processes.

```
python agent-stoat.py  →  Start (Interactive)
```

### 2. Persistent (+ Integrations)
Stoat runs continuously alongside you. It maintains long-term memory across sessions, fires background checks on a schedule, and can receive and respond to messages via Discord.

> Persistent mode is inspired by [OpenClaw](https://github.com/botanicastudios/openclaw), bringing that same always-on, memory-backed agent pattern to a fully local setup.

```
python agent-stoat.py  →  Start (Persistent + Integrations)
```

What this enables:
- **Long-term memory** — Stoat remembers facts, ongoing tasks, and user preferences across sessions in `MEMORY.md`
- **Daily log** — a rolling 4 AM→4 AM log of notable events, distilled into memory at reset
- **Heartbeat** — a background check every 30 minutes against a customizable checklist (`HEARTBEAT.md`)
- **Scheduler** — one-time and recurring tasks defined in `SCHEDULE.md`, checked every 2 minutes
- **Discord integration** — receive messages, send replies, route scheduled alerts to specific channels
- **4 AM daily reset** — daily log is consolidated into long-term memory, context is cleared for a fresh start

---

## Features

- **Streaming REPL** with colored output and command history
- **Tool calling** with robust multi-format parsing (native OpenAI, XML tags, markdown JSON, raw JSON)
- **Core tools**: file read/write/edit, shell execution, web fetch/search, directory listing, file/content search
- **Memory tools**: persistent memory scratchpad, daily log, schedule read/write
- **Per-tool permissions** — configure every tool as `allowed`, `denied`, or `ask`
- **Context compaction** — automatically summarizes old tool interactions to stay within context limits
- **Preset system** — switchable behavior configs (interactive, coder, persistent, with/without compaction)
- **Multi-backend** — CUDA (NVIDIA), Vulkan (AMD/cross-vendor), or CPU
- **Esc to interrupt** generation mid-stream
- **Startup menu** — configure model, context, permissions, preset, and backend before entering the REPL
- **Working directory** — start a new session folder or resume an existing one

---

## Requirements

- **Python 3.10+** (stdlib only — no pip packages required for core features)
- A **GGUF model file** (can be auto-downloaded on first run)

The **Vulkan backend** (AMD/NVIDIA/cross-vendor) is included in `llama_cpp_backends/vulkan/` and works out of the box.

The **CUDA backend** (NVIDIA, fastest) is too large to bundle. Download it separately:

1. Download [`llama-b8292-bin-win-cuda12-x64.zip`](https://github.com/ggml-org/llama.cpp/releases/tag/b8292) from the llama.cpp releases page
2. Extract the contents into `llama_cpp_backends/cuda/`

Linux users: download the appropriate build from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).

---

## Setup

1. Place a GGUF model file in the `models/` directory:
   ```
   agent-stoat/
     models/
       YourModel.gguf
   ```

2. Run the agent:
   ```
   python agent-stoat.py
   ```

On first run, the agent will:
- Auto-detect your GPU (CUDA > Vulkan > CPU fallback)
- Offer to download a default model from HuggingFace if none are found
- Start `llama-server` as a background process
- Create default state files in `.agent-stoat/` if they don't exist
- Clean up the server on exit

### Discord (Persistent mode only)

1. Create a Discord bot at [discord.com/developers](https://discord.com/developers/applications) and copy the token
2. Invite the bot to your server with `Send Messages` and `Add Reactions` permissions
3. Run `pip install discord.py`
4. In the startup menu, choose **Configure Discord** to enter your token and trigger mode

Discord trigger modes:
- `mention` — responds when @mentioned (default)
- `prefix` — responds to messages starting with `!stoat`
- `all` — responds to every message in channels it can read

### Updating llama.cpp binaries

Download from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and extract into the appropriate folder:
```
llama_cpp_backends/
  cuda/       ← llama-*-bin-win-cuda12-x64.zip  (download separately)
  vulkan/     ← llama-*-bin-win-vulkan-x64.zip  (included)
  cpu/        ← llama-*-bin-win-x64.zip
```

---

## Startup Menu

| Option | Description |
|---|---|
| **Start (Interactive)** | Basic mode — full tools, no background processes |
| **Start (Persistent + Integrations)** | Persistent mode — heartbeat, scheduler, Discord |
| **Load Model** | Pick from GGUF models in `models/` |
| **Set Context Window** | View/change context size (shows VRAM-based recommendations) |
| **Set Permissions** | Toggle any tool between `allowed`, `denied`, `ask` |
| **Switch Preset** | Choose behavior preset |
| **Switch Backend** | Switch between CUDA, Vulkan, CPU |
| **Configure Discord** | Enter bot token and trigger settings |

---

## Runtime Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/model <name>` | Switch model (restarts server) |
| `/models` | List local .gguf models |
| `/ctx <size>` | Set context window size |
| `/temp <value>` | Set temperature (0.0–2.0) |
| `/status` | Show current settings and context usage |
| `/permissions` | Show/set tool permissions (e.g. `/permissions shell y`) |
| `/presets` | List available presets |
| `/preset <name>` | Switch preset |
| `/backend <name>` | Switch GPU backend (cuda, vulkan, cpu) |
| `/compact [n]` | Manually compact history (keep last n messages) |
| `/clear` | Clear conversation history |
| `/heartbeat` | Trigger a heartbeat tick immediately (persistent mode) |
| `/schedule` | Trigger a schedule poll immediately (persistent mode) |
| `/exit` | Quit |
| `Esc` | Interrupt current generation |

---

## Tools

### Core Tools

| Tool | Description | Default |
|---|---|---|
| `read_file` | Read file contents (supports line ranges) | allowed |
| `write_file` | Create or overwrite a file | ask |
| `edit_file` | Replace a unique string in a file | ask |
| `shell` | Execute a shell command | ask |
| `list_dir` | List files and folders with sizes | allowed |
| `find_files` | Find files by glob pattern | allowed |
| `search_files` | Search for text/regex inside files | allowed |
| `web_fetch` | Fetch a URL and return clean text | allowed |
| `web_search` | Search the web via DuckDuckGo | allowed |

### Persistent Mode Tools

| Tool | Description |
|---|---|
| `read_memory` | Read long-term memory (`MEMORY.md`) |
| `update_memory` | Write to long-term memory |
| `read_daily_log` | Read the current 4 AM→4 AM activity log |
| `append_daily_log` | Append a timestamped entry to the daily log |
| `read_schedule` | Read `SCHEDULE.md` |
| `update_schedule` | Add or modify scheduled tasks |
| `get_current_time` | Get the current date and time |
| `clear_context` | Signal a context reset on the next interaction |

All tools can be individually configured to `allowed`, `denied`, or `ask` via the startup menu or `/permissions` command.

---

## Scheduling

Tasks are defined in `.agent-stoat/SCHEDULE.md`. Stoat checks it every 2 minutes.

**Formats:**
```
2026-04-01 09:00 | remind me to review the PR                  (one-time)
daily 08:30      | check if any Ongoing items in memory are due (daily)
every 4h         | summarize recent shell history              (interval)
every 30m        | check disk usage                            (interval)
```

**Direct messages** (no LLM, instant delivery):
```
2026-04-01 09:00 | message: <@USERID> Your reminder text. [ch:CHANNELID]
```

Use `read_schedule` and `update_schedule` tools, or edit the file directly.

---

## Memory

In persistent mode, Stoat maintains two memory layers:

**Long-term memory** (`MEMORY.md`) — structured scratchpad with four sections:
```markdown
## People    — facts about users: preferences, timezone, context
## Facts     — permanent context worth keeping indefinitely
## Ongoing   — active tasks, follow-ups, things in progress
## Notes     — temporary items, cleaned up periodically
```

**Daily log** (`DAILY_LOG.md`) — an append-only log covering the current 4 AM→4 AM window. At 4 AM each day, a scheduled task automatically:
1. Reads the daily log
2. Distills worth-keeping entries into the structured memory
3. Clears the log
4. Resets conversation context for a fresh start

**Character** (`SOUL.md`) — optional personality file loaded into every system prompt. Edit it to customize Stoat's tone and behavior. Created with a default on first run.

---

## Persistent Mode Internals

Four concurrent threads run in persistent mode, all sharing a single `_turn_lock` so LLM calls never overlap:

| Thread | Purpose |
|---|---|
| **Main** | REPL — reads CLI input, drains Discord message queue |
| **Discord bot** | Asyncio event loop receiving Discord events |
| **Heartbeat** | Fires every 30 minutes, checks `HEARTBEAT.md` |
| **Scheduler** | Polls `SCHEDULE.md` every 2 minutes, fires due tasks |

Discord conversations are isolated per server (each guild has its own conversation history). CLI and Discord histories never mix.

---

## Project Structure

```
agent-stoat/
  agent-stoat.py                  # Entry point
  agent-stoat_scripts/
    config.py                     # Model and generation defaults
    llm_server.py                 # llama-server lifecycle (GPU detection, start/stop)
    llm_client.py                 # OpenAI-compatible streaming API client
    chat_engine.py                # Core chat loop with tool calling and compaction
    tool_parser.py                # Multi-format tool call extraction
    tools.py                      # Tool definitions, implementations, permission system
    persistent_mode.py            # Heartbeat background loop
    schedule_runner.py            # Cron-style task scheduler
    integrations.py               # Optional package management (discord.py, etc.)
    discord_bridge.py             # Discord bot integration
    prompt_basic.md               # System prompt — interactive mode
    prompt_coder.md               # System prompt — coding-focused variant
    prompt_persistent.md          # System prompt — persistent + integrations mode
    prompt_heartbeat.md           # System prompt — background heartbeat ticks
    agent-stoat.png               # Logo
  llama_cpp_backends/
    cuda/                         # Download separately (see Setup)
    vulkan/                       # Included — works out of the box
  models/                         # GGUF model files (gitignored, user-provided)
  agent-stoat_working-dir/        # Session working directories (gitignored)
  .agent-stoat/
    presets/                      # Behavior preset JSON files (tracked)
    settings.json                 # User settings and Discord config (gitignored)
    settings.json.example         # Template — copy to settings.json to configure
    MEMORY.md                     # Long-term memory scratchpad (gitignored)
    SOUL.md                       # Character/personality definition (gitignored)
    SCHEDULE.md                   # Scheduled tasks (gitignored)
    HEARTBEAT.md                  # Heartbeat checklist (gitignored)
    DAILY_LOG.md                  # Rolling daily activity log (gitignored)
```

---

## Configuration

Edit `config.py` to change defaults, or use `.agent-stoat/settings.json` for runtime overrides (see `settings.json.example`):

| Setting | Default | Description |
|---|---|---|
| `MODEL_REPO` | `unsloth/Qwen3.5-35B-A3B-GGUF` | HuggingFace repo for auto-download |
| `MODEL_FILE` | `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` | Default GGUF filename |
| `TEMPERATURE` | `0.7` | Generation temperature |
| `NUM_CTX` | `32768` | Context window size in tokens |
| `MAX_ITERATIONS` | `50` | Max tool-calling rounds per user message |

---

## License

MIT License. See [LICENSE](LICENSE).
