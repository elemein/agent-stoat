"""integrations.py — optional third-party package management for Agent Stoat.

Handles install-checking and pip-install for optional integrations (e.g. discord.py),
and reads/writes integration settings in .agent-stoat/settings.json.
"""

import importlib
import json
import os
import subprocess
import sys


def is_installed(package_import_name: str) -> bool:
    """Return True if a package can be imported (i.e. is installed)."""
    try:
        importlib.import_module(package_import_name)
        return True
    except ImportError:
        return False


def get_package_version(package_import_name: str) -> str | None:
    """Return the installed version string, or None if not installed."""
    try:
        mod = importlib.import_module(package_import_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def install_package(pip_name: str) -> bool:
    """Run pip install <pip_name> and stream output to stdout. Returns True on success."""
    print(f"\n  Installing {pip_name}...")
    try:
        _win_flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            **_win_flags
        )
        for line in result.stdout.splitlines():
            print(f"    {line}")
        if result.returncode == 0:
            print(f"  {pip_name} installed successfully.")
            return True
        else:
            print(f"  Installation failed (exit code {result.returncode}).")
            return False
    except Exception as e:
        print(f"  Installation error: {e}")
        return False


# ---------------------------------------------------------------------------
# Discord settings — stored under "discord" key in .agent-stoat/settings.json
# ---------------------------------------------------------------------------

_SETTINGS_PATH_REF = None  # Set by agent-stoat.py after AGENT_DATA_DIR is known


def _settings_path() -> str:
    """Resolve the path to .agent-stoat/settings.json via the tools module."""
    from tools import AGENT_DATA_DIR
    return os.path.join(AGENT_DATA_DIR, "settings.json")


def _load_settings_file() -> dict:
    path = _settings_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings_file(data: dict) -> None:
    path = _settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_discord_settings() -> dict:
    """Return the discord settings block, with defaults filled in."""
    data = _load_settings_file()
    defaults = {
        "token": "",
        "trigger": "mention",   # "all", "mention", "prefix"
        "prefix": "!stoat",
    }
    stored = data.get("discord", {})
    return {**defaults, **stored}


def save_discord_settings(discord_cfg: dict) -> None:
    """Merge discord_cfg into the settings file under the 'discord' key."""
    data = _load_settings_file()
    data["discord"] = discord_cfg
    _save_settings_file(data)
