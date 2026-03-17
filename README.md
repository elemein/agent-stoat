# Agent Stoat

![Stoat](agent-stoat_scripts/agent-stoat.png)

Local-first coding agent built with small models in mind and tight VRAM budgets. Powered by llama.cpp.

## Features

- **Streaming REPL** with colored output and command history
- **Tool calling** with robust multi-format parsing (native OpenAI-compatible, XML tags, markdown JSON, raw JSON)
- **Core tools**: file read/write/edit, shell execution, web fetch/search, directory listing, file/content search
- **Per-tool permissions** — configure every tool as `allowed`, `denied`, or `ask` (prompt each time)
- **Context compaction** — automatically summarizes old tool interactions to preserve context over long conversations
- **Preset system** — switchable behavior configs (with or without compaction)
- **Multi-backend** — CUDA (NVIDIA), Vulkan (AMD/cross-vendor), or CPU
- **Esc to interrupt** generation mid-stream
- **Startup menu** — configure model, context size, permissions, preset, and backend before entering the REPL
- **Working directory selection** — start a new session folder or resume an existing one
- **Runtime configuration** — switch models, adjust context size, temperature, backend on the fly

## Requirements

- **Python 3.10+** (no pip packages required — uses only the standard library)
- A **GGUF model file** (can be downloaded on first run)

The **Vulkan backend** (AMD/NVIDIA/cross-vendor) is included in `llama_cpp_backends/vulkan/` and works out of the box.

The **CUDA backend** (NVIDIA, fastest) is too large to bundle in the repo. Download it separately:

1. Download [`llama-b8292-bin-win-cuda12-x64.zip`](https://github.com/ggml-org/llama.cpp/releases/tag/b8292) from the llama.cpp releases page
2. Extract the contents into `llama_cpp_backends/cuda/`

Linux users will need to download the appropriate build from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).

## Setup

1. Place a GGUF model file in the `models/` directory:
   ```
   agent-stoat/
     models/
       YourModel.gguf
   ```

2. Run the agent:
   ```bash
   python agent-stoat.py
   ```

On first run, the agent will:
- Auto-detect your GPU (CUDA > Vulkan > CPU fallback)
- Offer to download a default model from HuggingFace if none are found
- Start `llama-server` as a background process
- Clean up the server on exit

Two directories are created automatically:
- `.agent-stoat/` — runtime state (presets, server logs, settings)
- `agent-stoat_working-dir/` — session folders where the agent does its work

### Updating llama.cpp binaries

To update backends, download from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and extract into the appropriate folder:
```
llama_cpp_backends/
  cuda/       ← llama-*-bin-win-cuda12-x64.zip  (download separately)
  vulkan/     ← llama-*-bin-win-vulkan-x64.zip  (included)
  cpu/        ← llama-*-bin-win-x64.zip
```

## Startup Menu

On launch, an interactive menu lets you configure the agent:

- **Start** — press Enter or `1` to go straight into the REPL
- **Load Model** — pick from GGUF models in `models/`
- **Set Context Window** — view/change context size with VRAM-based recommendations
- **Set Permissions** — toggle any tool between `allowed`, `denied`, and `ask`
- **Switch Preset** — choose between behavior presets (e.g. basic, basic with compaction)
- **Switch Backend** — switch between CUDA, Vulkan, and CPU

## Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/model <name>` | Switch model (restarts server) |
| `/models` | List local .gguf models |
| `/ctx <size>` | Set context window size |
| `/temp <value>` | Set temperature (0.0 - 2.0) |
| `/status` | Show current settings and context usage |
| `/permissions` | Show/set tool permissions (e.g. `/permissions shell y`) |
| `/presets` | List available presets |
| `/preset <name>` | Switch preset |
| `/backend <name>` | Switch GPU backend (cuda, vulkan, cpu) |
| `/compact [n]` | Manually compact history (keep last n messages) |
| `/clear` | Clear conversation history |
| `/exit` | Quit |
| `Esc` | Interrupt current generation |

## Tools

| Tool | Description | Permission Default |
|---|---|---|
| `read_file` | Read file contents (5000 char limit, supports line ranges) | allowed |
| `write_file` | Create or overwrite a file | ask |
| `edit_file` | Replace a unique string in a file | ask |
| `shell` | Execute a shell command (3000 char output limit) | ask |
| `list_dir` | List files and folders with sizes (50 entries) | allowed |
| `find_files` | Find files by glob pattern recursively (30 results) | allowed |
| `search_files` | Search for text/regex inside files (30 matches) | allowed |
| `web_fetch` | Fetch a URL and return clean text (3000 chars) | allowed |
| `web_search` | Search the web via DuckDuckGo (5 results) | allowed |

All tools can be individually set to `allowed`, `denied`, or `ask` via the startup menu or `/permissions` command.

## Project Structure

```
agent-stoat/
  agent-stoat.py                  # Entry point
  agent-stoat_scripts/
    config.py               # Model and generation defaults
    llm_server.py           # llama-server lifecycle (GPU detection, download, start/stop)
    llm_client.py           # OpenAI-compatible streaming API client
    chat_engine.py          # Core chat loop with tool calling and context compaction
    tool_parser.py          # Multi-format tool call extraction from LLM responses
    tools.py                # Tool definitions, implementations, and permission system
    prompt_basic.md         # System prompt
    agent-stoat.png         # Logo
  llama_cpp_backends/
    cuda/               # Empty — download CUDA build separately (see README)
    vulkan/             # Included — works out of the box
  models/                   # GGUF model files (gitignored)
  agent-stoat_working-dir/  # Session working directories (gitignored)
  .agent-stoat/             # Runtime state (gitignored)
    presets/                # Behavior preset JSON files
    settings.json           # User settings overlay
    llama-server.log        # Server output log
```

## Configuration

Edit `config.py` to change defaults, or use `.agent-stoat/settings.json` for overrides:

| Setting | Default | Description |
|---|---|---|
| `MODEL_REPO` | `unsloth/Qwen3.5-35B-A3B-GGUF` | HuggingFace repo for auto-download |
| `MODEL_FILE` | `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` | Default GGUF model filename |
| `TEMPERATURE` | `0.7` | Generation temperature |
| `NUM_CTX` | `32768` | Context window size in tokens |
| `MAX_ITERATIONS` | `50` | Max tool-calling rounds per user message |

Settings overlay format (`.agent-stoat/settings.json`):
```json
{
  "model": "YourModel.gguf",
  "temperature": 0.7,
  "num_ctx": 16384,
  "permissions": {
    "shell": "allow",
    "web_fetch": "deny"
  }
}
```

## License

MIT License. See [LICENSE](LICENSE).
