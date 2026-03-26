"""llama.cpp server lifecycle manager — GPU detection, model download, server spawn."""

import atexit
import os
import platform
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json
import shutil

# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
_BACKENDS_DIR = os.path.join(_PROJECT_DIR, "llama_cpp_backends")  # backend subdirs (cuda/, vulkan/, cpu/)
_MODELS_DIR = os.path.join(_PROJECT_DIR, "models")         # GGUF model files

IS_WINDOWS = platform.system() == "Windows"

# Default server port
SERVER_PORT = 8080
SERVER_HOST = f"http://localhost:{SERVER_PORT}"

# ── GPU Detection ─────────────────────────────────────────────────────────────

def detect_gpu() -> str:
    """Detect available GPU backend. Returns 'cuda', 'vulkan', or 'cpu'."""

    _win_flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WINDOWS else {}

    # Check NVIDIA first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, **_win_flags
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0]
            print(f"  \033[32m✓\033[0m Detected NVIDIA GPU: {gpu_name}")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check Vulkan
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"] if not IS_WINDOWS else ["vulkaninfo.exe", "--summary"],
            capture_output=True, text=True, timeout=5, **_win_flags
        )
        if result.returncode == 0:
            print("  \033[32m✓\033[0m Detected Vulkan-capable GPU")
            return "vulkan"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("  \033[33m⚠\033[0m No GPU detected — using CPU backend")
    return "cpu"


def get_vram_info() -> dict:
    """Query nvidia-smi for GPU VRAM. Returns {} if unavailable."""
    try:
        _win_flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if IS_WINDOWS else {}
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, **_win_flags
        )
        if result.returncode != 0:
            return {}
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return {}
        total_mb = free_mb = 0
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 2:
                total_mb += int(parts[0].strip())
                free_mb += int(parts[1].strip())
        return {"total_mb": total_mb, "free_mb": free_mb, "gpu_count": len(lines)}
    except Exception:
        return {}


# ── Server Binary ─────────────────────────────────────────────────────────────

def _server_binary_path(backend: str) -> str:
    """Get path to llama-server binary for the given backend."""
    exe = "llama-server.exe" if IS_WINDOWS else "llama-server"
    backend_dir = os.path.join(_BACKENDS_DIR, backend)
    if os.path.isdir(backend_dir):
        path = os.path.join(backend_dir, exe)
        if os.path.isfile(path):
            return path
    return ""


def find_server_binary(backend: str) -> str:
    """Find the llama-server binary, or prompt for setup."""
    path = _server_binary_path(backend)
    if path:
        return path

    print(f"\n  \033[31m✗\033[0m llama-server binary not found for backend '{backend}'")
    print(f"    Expected location: llama_cpp_backends/{backend}/")
    print()
    print("    Setup instructions:")
    print(f"    1. Download the prebuilt release for your platform from:")
    print(f"       https://github.com/ggml-org/llama.cpp/releases")
    if IS_WINDOWS:
        if backend == "cuda":
            print(f"       → Look for: llama-*-bin-win-cuda12-x64.zip")
        elif backend == "vulkan":
            print(f"       → Look for: llama-*-bin-win-vulkan-x64.zip")
        else:
            print(f"       → Look for: llama-*-bin-win-x64.zip (CPU)")
    else:
        print(f"       → Look for the Linux x64 build matching your backend")
    print(f"    2. Extract llama-server{'.exe' if IS_WINDOWS else ''} (and any DLLs) into:")
    print(f"       {os.path.join(_BACKENDS_DIR, backend)}/")
    print()
    return ""


# ── Model Management ──────────────────────────────────────────────────────────

def find_model(model_filename: str) -> str:
    """Find a GGUF model file. Returns path or empty string."""
    path = os.path.join(_MODELS_DIR, model_filename)
    if os.path.isfile(path):
        return path
    # Also check if the full path was given
    if os.path.isfile(model_filename):
        return model_filename
    return ""


def read_gguf_metadata(model_path: str) -> dict:
    """Read metadata from a GGUF file header.

    Only parses the keys we care about — skips large arrays (tokenizer vocab etc.)
    to avoid reading megabytes of data from the header.
    """
    # Keys we actually need
    _WANTED_SUFFIXES = {
        "general.architecture", "general.name", "general.file_type",
        "general.parameter_count",
        ".context_length", ".block_count",
        ".attention.head_count", ".attention.head_count_kv",
        ".embedding_length",
    }

    def _key_wanted(key):
        for suffix in _WANTED_SUFFIXES:
            if key == suffix or key.endswith(suffix):
                return True
        return False

    # Size in bytes for fixed-width types
    _TYPE_SIZE = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
        10: 8, 11: 8, 12: 8,
    }

    def _read_string(f):
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")

    def _read_value(f, vtype):
        if vtype in _TYPE_SIZE:
            fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<B",10:"<Q",11:"<q",12:"<d"}[vtype]
            val = struct.unpack(fmt, f.read(_TYPE_SIZE[vtype]))[0]
            return bool(val) if vtype == 7 else val
        elif vtype == 8:
            return _read_string(f)
        elif vtype == 9:  # array
            return None  # handled by caller
        return None

    def _skip_value(f, vtype):
        """Skip over a value without parsing it."""
        if vtype in _TYPE_SIZE:
            f.seek(_TYPE_SIZE[vtype], 1)
        elif vtype == 8:
            length = struct.unpack("<Q", f.read(8))[0]
            f.seek(length, 1)
        elif vtype == 9:  # array — skip all elements
            elem_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            if elem_type in _TYPE_SIZE:
                f.seek(_TYPE_SIZE[elem_type] * count, 1)
            elif elem_type == 8:
                for _ in range(count):
                    sl = struct.unpack("<Q", f.read(8))[0]
                    f.seek(sl, 1)

    result = {}
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return {}

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            metadata = {}
            for _ in range(kv_count):
                key = _read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]

                if _key_wanted(key) and vtype != 9:
                    metadata[key] = _read_value(f, vtype)
                else:
                    _skip_value(f, vtype)

        # Extract architecture name (e.g., "qwen2", "llama")
        arch = metadata.get("general.architecture", "")
        result["architecture"] = arch
        result["model_name"] = metadata.get("general.name", "")
        result["context_length"] = (
            metadata.get(f"{arch}.context_length")
            or metadata.get("llama.context_length")
        )
        result["block_count"] = metadata.get(f"{arch}.block_count")
        result["head_count"] = metadata.get(f"{arch}.attention.head_count")
        result["head_count_kv"] = metadata.get(f"{arch}.attention.head_count_kv")
        result["embedding_length"] = metadata.get(f"{arch}.embedding_length")
        result["parameter_count"] = metadata.get("general.parameter_count")

        FILE_TYPE_NAMES = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
            8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
            13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
            18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 24: "IQ4_XS",
        }
        ft = metadata.get("general.file_type")
        result["quantization"] = FILE_TYPE_NAMES.get(ft, f"type_{ft}") if isinstance(ft, int) else ""

    except Exception as e:
        result["_error"] = str(e)

    return result


# Cache parsed metadata to avoid re-reading the file
_gguf_metadata_cache: dict[str, dict] = {}


def get_gguf_metadata(model_path: str) -> dict:
    """Get GGUF metadata (cached)."""
    if model_path not in _gguf_metadata_cache:
        _gguf_metadata_cache[model_path] = read_gguf_metadata(model_path)
    return _gguf_metadata_cache[model_path]


def download_model(repo: str, filename: str, dest_path: str) -> bool:
    """Download a GGUF model from HuggingFace.

    Args:
        repo: HuggingFace repo (e.g. "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF")
        filename: GGUF filename (e.g. "Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf")
        dest_path: Full local path to save to
    """
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"  Downloading {filename}...")
    print(f"  From: {repo}")
    print(f"  URL:  {url}")
    print()

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agent-stoat/1.0"})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            total_gb = total / (1024 ** 3) if total else 0
            tmp_path = dest_path + ".part"

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        pct = downloaded / total * 100
                        dl_gb = downloaded / (1024 ** 3)
                        bar_w = 30
                        filled = int(bar_w * pct / 100)
                        bar = "█" * filled + "░" * (bar_w - filled)
                        sys.stdout.write(
                            f"\r  [{bar}] {pct:5.1f}% ({dl_gb:.1f}/{total_gb:.1f} GB)"
                        )
                        sys.stdout.flush()

            # Rename .part to final name
            shutil.move(tmp_path, dest_path)
            print(f"\n  \033[32m✓\033[0m Download complete: {dest_path}")
            return True

    except Exception as e:
        print(f"\n  \033[31m✗\033[0m Download failed: {e}")
        # Cleanup partial download
        tmp_path = dest_path + ".part"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


# ── Server Lifecycle ──────────────────────────────────────────────────────────

_server_process = None


def start_server(
    binary_path: str,
    model_path: str,
    port: int = SERVER_PORT,
    n_gpu_layers: int = 99,
    ctx_size: int = 8192,
    extra_args: list = None,
) -> bool:
    """Start llama-server as a subprocess.

    Returns True if server started and is healthy.
    """
    global _server_process, SERVER_HOST, SERVER_PORT

    SERVER_PORT = port
    SERVER_HOST = f"http://localhost:{port}"

    cmd = [
        binary_path,
        "-m", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size),
        "--jinja",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Starting llama-server on port {port}...")
    print(f"  Model: {os.path.basename(model_path)}")
    print(f"  GPU layers: {n_gpu_layers}, Context: {ctx_size}")

    try:
        # Redirect server stdout/stderr to log file
        log_path = os.path.join(_PROJECT_DIR, ".agent-stoat", "llama-server.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w")

        # Set environment — inherit current env plus backend-specific fixes
        env = os.environ.copy()
        if _env_backend == "vulkan":
            # Workaround for NVIDIA Windows driver bug: Vulkan host-visible
            # video memory isn't freed on process exit, causing BSOD.
            env["GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"] = "1"

        kwargs = {
            "stdout": log_file,
            "stderr": log_file,
            "cwd": os.path.dirname(binary_path),  # DLLs live next to the binary
            "env": env,
        }
        # On Windows, create a new process group so we can kill it cleanly,
        # and suppress the console window that would otherwise flash up.
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

        _server_process = subprocess.Popen(cmd, **kwargs)

        # Register cleanup
        atexit.register(stop_server)

        # Give it a moment, then check if it died immediately
        time.sleep(2)
        exit_code = _server_process.poll()
        if exit_code is not None:
            log_file.close()
            log_content = ""
            try:
                with open(log_path, "r") as f:
                    log_content = f.read().strip()
            except Exception:
                pass
            print(f"  \033[31m✗\033[0m Server exited immediately (code {exit_code})")
            if log_content:
                for line in log_content.splitlines()[:10]:
                    print(f"    {line}")
            else:
                print(f"    No log output. The binary may be incompatible or missing DLLs.")
                print(f"    Try running manually:")
                print(f"      cd {os.path.dirname(binary_path)}")
                print(f"      .\\{os.path.basename(binary_path)} -m \"{model_path}\" --port {port}")
            _server_process = None
            return False

        # Wait for server to be healthy
        if _wait_for_health(timeout=120):
            print(f"  \033[32m✓\033[0m Server ready on port {port}")
            return True
        else:
            print(f"  \033[31m✗\033[0m Server failed to start within 120s")
            print(f"    Check log: {log_path}")
            stop_server()
            return False

    except FileNotFoundError:
        print(f"  \033[31m✗\033[0m Binary not found: {binary_path}")
        return False
    except Exception as e:
        print(f"  \033[31m✗\033[0m Failed to start server: {e}")
        return False


def _wait_for_health(timeout: int = 60) -> bool:
    """Poll the server's /health endpoint until it's ready."""
    url = f"{SERVER_HOST}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check if process died
        if _server_process and _server_process.poll() is not None:
            return False

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read())
                status = data.get("status", "")
                if status == "ok":
                    return True
                # "loading model" — keep waiting
                sys.stdout.write(f"\r  Loading model... ({status})")
                sys.stdout.flush()
        except (urllib.error.URLError, ConnectionRefusedError, OSError, Exception):
            pass

        time.sleep(1)

    print()
    return False


def stop_server():
    """Stop the llama-server subprocess."""
    global _server_process
    if _server_process is None:
        return

    try:
        if _server_process.poll() is not None:
            _server_process = None
            return

        _server_process.terminate()
        try:
            _server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_process.kill()
            _server_process.wait(timeout=5)
    except Exception:
        try:
            _server_process.kill()
        except Exception:
            pass

    _server_process = None


def is_server_running() -> bool:
    """Check if the server process is alive and healthy."""
    if _server_process is None or _server_process.poll() is not None:
        return False
    try:
        req = urllib.request.Request(f"{SERVER_HOST}/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except Exception:
        return False


# ── High-Level Setup ──────────────────────────────────────────────────────────

# Cached environment state (set by setup_environment)
_env_backend = None
_env_binary = None


def get_backend() -> str:
    """Return the detected GPU backend ('cuda', 'vulkan', or 'cpu')."""
    return _env_backend or "unknown"


def set_backend(backend: str) -> bool:
    """Switch to a different backend. Returns True if the binary was found.

    If the server is currently running, it must be restarted for changes to
    take effect (use load_model again).
    """
    global _env_backend, _env_binary
    backend = backend.lower()
    if backend not in ("cuda", "vulkan", "cpu"):
        return False
    binary = find_server_binary(backend)
    if not binary:
        return False
    _env_backend = backend
    _env_binary = binary
    return True


def list_local_models() -> list[str]:
    """List .gguf files in the models/ directory. Returns filenames sorted by name."""
    os.makedirs(_MODELS_DIR, exist_ok=True)
    return sorted(
        f for f in os.listdir(_MODELS_DIR)
        if f.lower().endswith(".gguf") and not f.endswith(".part")
    )


def get_models_dir() -> str:
    """Return the models directory path."""
    return _MODELS_DIR


def setup_environment(port: int = SERVER_PORT) -> bool:
    """Detect GPU and find llama-server binary. Does NOT start the server or require a model.

    Returns True if the environment is ready (binary found).
    Call load_model() afterwards to start serving a specific model.
    """
    global _env_backend, _env_binary, SERVER_PORT, SERVER_HOST

    SERVER_PORT = port
    SERVER_HOST = f"http://localhost:{port}"

    print()
    print("  \033[1m── Inference Server Setup ──\033[0m")
    print()

    # 1. Detect GPU
    _env_backend = detect_gpu()

    # 2. Find server binary
    _env_binary = find_server_binary(_env_backend)
    if not _env_binary:
        # Detected cuda but only vulkan binary exists (or vice versa) — try fallbacks
        for fallback in ("cuda", "vulkan", "cpu"):
            if fallback != _env_backend:
                _env_binary = _server_binary_path(fallback)
                if _env_binary:
                    print(f"  \033[33m⚠\033[0m No {_env_backend} binary found, falling back to {fallback}")
                    _env_backend = fallback
                    break
        if not _env_binary:
            return False

    print(f"  \033[32m✓\033[0m Server binary ready ({_env_backend.upper()} backend)")
    local_models = list_local_models()
    if local_models:
        print(f"  \033[32m✓\033[0m {len(local_models)} model(s) in models/")
    else:
        print(f"  \033[33m⚠\033[0m No models found in models/")
    print()
    return True


def load_model(
    model_file: str,
    ctx_size: int = 8192,
    model_repo: str = None,
) -> bool:
    """Start (or restart) the server with a specific model.

    If the model file isn't found locally and model_repo is provided,
    offers to download it. Returns True if the server is running.
    """
    if not _env_binary:
        print("  \033[31m✗\033[0m Environment not set up. Call setup_environment() first.")
        return False

    # Stop existing server if running
    if is_server_running():
        print(f"  Stopping current server...")
        stop_server()

    # Find the model
    model_path = find_model(model_file)
    if not model_path:
        if model_repo:
            print(f"\n  Model not found: {model_file}")
            dest = os.path.join(_MODELS_DIR, model_file)
            answer = input("  Download it now? [Y/n] ").strip().lower()
            if answer in ("", "y", "yes"):
                if not download_model(model_repo, model_file, dest):
                    return False
                model_path = dest
            else:
                print("  Aborted. Place GGUF files in:")
                print(f"    {_MODELS_DIR}/")
                return False
        else:
            print(f"  \033[31m✗\033[0m Model not found: {model_file}")
            print(f"    Place GGUF files in: {_MODELS_DIR}/")
            return False

    n_gpu_layers = 99 if _env_backend != "cpu" else 0

    return start_server(
        binary_path=_env_binary,
        model_path=model_path,
        port=SERVER_PORT,
        n_gpu_layers=n_gpu_layers,
        ctx_size=ctx_size,
    )


