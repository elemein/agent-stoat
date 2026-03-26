"""discord_bridge.py — Discord bot bridge for Agent Stoat.

Runs a discord.py bot in a background thread alongside the CLI REPL.
Incoming Discord messages are placed on a shared queue; the main REPL
loop drains that queue between (or instead of) CLI inputs and calls the
same conversational_turn() / run_chat() path.

Requires discord.py:  pip install discord.py
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Callable


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class DiscordMessage:
    """Wrapper for an incoming Discord message routed to Stoat."""

    def __init__(
        self,
        author: str,
        user_id: int,
        user_mention: str,
        content: str,
        channel_id: int,
        guild_id: int | None,
        guild_name: str,
        reply_fn: Callable[[str], None],
    ):
        self.author = author
        self.user_id = user_id
        self.user_mention = user_mention  # "<@123456789>" — for @-ing back in permission prompts
        self.content = content
        self.channel_id = channel_id
        self.guild_id = guild_id          # None for DMs
        self.guild_name = guild_name      # Server name, or "DM" for direct messages
        self.reply = reply_fn  # call reply("text") to send back to Discord


# ---------------------------------------------------------------------------
# Bridge state
# ---------------------------------------------------------------------------

_bot_thread: threading.Thread | None = None
_bot_loop: asyncio.AbstractEventLoop | None = None
_bot_client = None
_message_queue: queue.Queue[DiscordMessage] = queue.Queue()
_interrupt_event = threading.Event()  # set to interrupt current Discord-sourced turn
_running = False

_worker_thread: threading.Thread | None = None
_worker_stop = threading.Event()

# Most recently active Discord channel object — used as fallback for alerts.
# Storing the object (not just the ID) means DM channels work without a fetch.
_last_active_channel = None


def get_message_queue() -> queue.Queue[DiscordMessage]:
    return _message_queue


def is_running() -> bool:
    return _running


def interrupt_current_turn() -> None:
    """Signal the current Discord-sourced turn to stop (mirrors Esc in CLI)."""
    _interrupt_event.set()


def clear_interrupt() -> None:
    _interrupt_event.clear()


def is_interrupted() -> bool:
    return _interrupt_event.is_set()


# ---------------------------------------------------------------------------
# Permission requests
# ---------------------------------------------------------------------------

PERMISSION_TIMEOUT = 60.0  # seconds to wait for a reaction before auto-denying


def request_permission(
    channel_id: int,
    user_id: int,
    user_mention: str,
    tool_name: str,
    args: dict,
) -> bool:
    """Post a permission prompt to Discord and block until the user reacts.

    Posts a message @-ing the user with ✅ / ❌ reactions. Waits up to
    PERMISSION_TIMEOUT seconds for the original user to react. Returns True
    (allow) or False (deny / timeout).
    """
    if not _running or _bot_client is None or _bot_loop is None:
        return False

    result_holder: list[bool] = []
    result_event = threading.Event()

    async def _ask():
        try:
            channel = _bot_client.get_channel(channel_id)
            if channel is None:
                channel = await _bot_client.fetch_channel(channel_id)

            # Format args for display (truncate long values)
            if args:
                args_lines = "\n".join(
                    f"> `{k}`: {str(v)[:80]}{'…' if len(str(v)) > 80 else ''}"
                    for k, v in args.items()
                )
                args_block = f"\n{args_lines}"
            else:
                args_block = ""

            prompt = (
                f"{user_mention} Stoat wants to run **`{tool_name}`**{args_block}\n\n"
                f"React ✅ to **allow** or ❌ to **deny**"
            )
            msg = await channel.send(prompt)
            await msg.add_reaction("✅")
            await msg.add_reaction("❌")

            def _check(reaction, user):
                return (
                    user.id == user_id
                    and reaction.message.id == msg.id
                    and str(reaction.emoji) in ("✅", "❌")
                )

            try:
                reaction, _ = await asyncio.wait_for(
                    _bot_client.wait_for("reaction_add", check=_check),
                    timeout=PERMISSION_TIMEOUT,
                )
                allowed = str(reaction.emoji) == "✅"
                result_holder.append(allowed)
                status = "✅ Allowed" if allowed else "❌ Denied"
                await msg.edit(content=prompt + f"\n\n**{status}**")
            except asyncio.TimeoutError:
                result_holder.append(False)
                await msg.edit(content=prompt + "\n\n**⏱️ Timed out — denied**")

        except Exception as e:
            print(f"\n  \033[31m[Discord] Permission request error: {e}\033[0m")
            result_holder.append(False)
        finally:
            result_event.set()

    asyncio.run_coroutine_threadsafe(_ask(), _bot_loop)
    result_event.wait(timeout=PERMISSION_TIMEOUT + 10)
    return result_holder[0] if result_holder else False


# ---------------------------------------------------------------------------
# Discord bot
# ---------------------------------------------------------------------------

def _make_client(cfg: dict):
    """Build and return a configured discord.Client instance."""
    import discord  # noqa: deferred import — only called if discord.py is installed

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    trigger: str = cfg.get("trigger", "mention")
    prefix: str = cfg.get("prefix", "!stoat").strip()

    # ---- chunk helper ----
    def _chunks(text: str, size: int = 1900) -> list[str]:
        """Split text into Discord-safe chunks (≤ size chars)."""
        lines = text.split("\n")
        parts, current = [], ""
        for line in lines:
            if len(current) + len(line) + 1 > size:
                if current:
                    parts.append(current)
                current = line
            else:
                current = (current + "\n" + line).lstrip("\n")
        if current:
            parts.append(current)
        return parts or [""]

    # ---- event handlers ----
    @client.event
    async def on_ready():
        print(f"\n  \033[32m[Discord] Logged in as {client.user} (ID {client.user.id})\033[0m")
        print(f"  \033[2m[Discord] Trigger: {trigger}"
              f"{' | prefix: ' + prefix if trigger == 'prefix' else ''}\033[0m")

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        content = message.content.strip()

        # Trigger filter
        if trigger == "mention":
            if client.user not in message.mentions:
                return
            content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()
        elif trigger == "prefix":
            if not content.lower().startswith(prefix.lower()):
                return
            content = content[len(prefix):].strip()
        # trigger == "all": accept everything

        if not content:
            return

        reply_text_holder: list[str] = []

        def sync_reply(text: str):
            """Called from the main thread once the response is complete."""
            reply_text_holder.append(text)

        dm = DiscordMessage(
            author=message.author.display_name,
            user_id=message.author.id,
            user_mention=message.author.mention,
            content=content,
            channel_id=message.channel.id,
            guild_id=message.guild.id if message.guild else None,
            guild_name=message.guild.name if message.guild else "DM",
            reply_fn=sync_reply,
        )
        global _last_active_channel
        _last_active_channel = message.channel
        _message_queue.put(dm)

        # Show typing while the main thread processes, then send reply.
        # Time out after 180 s so a hung turn doesn't leave the indicator running forever.
        async def _typing_and_reply():
            async with message.channel.typing():
                waited = 0.0
                while not reply_text_holder:
                    await asyncio.sleep(0.1)
                    waited += 0.1
                    if waited >= 180.0:
                        await message.channel.send("_(no response — the turn may have timed out)_")
                        return
            for chunk in _chunks(reply_text_holder[0]):
                await message.channel.send(chunk)

        asyncio.ensure_future(_typing_and_reply())

    @client.event
    async def on_reaction_add(reaction, user):
        """❌ reaction on a bot message interrupts the current turn.

        Only fires when there is no active permission prompt — permission
        prompts have their own wait_for listener that handles the reaction.
        """
        if user == client.user:
            return
        if reaction.message is None:
            return
        if str(reaction.emoji) == "❌" and reaction.message.author == client.user:
            # Check whether the message text looks like a permission prompt;
            # if so, the wait_for in request_permission handles it instead.
            if "React ✅ to **allow**" not in (reaction.message.content or ""):
                _interrupt_event.set()

    return client


def start(cfg: dict) -> bool:
    """Start the Discord bot in a background thread.

    Returns True if started successfully, False on error.
    """
    global _bot_thread, _bot_loop, _bot_client, _running

    if _running:
        return True

    token = cfg.get("token", "").strip()
    if not token:
        print("  \033[31m[Discord] No bot token configured. Set it in Integrations → Discord Settings.\033[0m")
        return False

    try:
        import discord  # noqa
    except ImportError:
        print("  \033[31m[Discord] discord.py not installed. Go to Integrations to install it.\033[0m")
        return False

    _bot_client = _make_client(cfg)

    def _run_loop():
        global _bot_loop, _running
        _bot_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_bot_loop)
        try:
            _running = True
            _bot_loop.run_until_complete(_bot_client.start(token))
        except Exception as e:
            print(f"\n  \033[31m[Discord] Bot error: {e}\033[0m")
        finally:
            _running = False

    _bot_thread = threading.Thread(target=_run_loop, daemon=True, name="discord-bot")
    _bot_thread.start()
    return True


def stop() -> None:
    """Stop the Discord bot gracefully."""
    global _running, _bot_client, _bot_loop

    if not _running or _bot_client is None:
        return

    if _bot_loop and not _bot_loop.is_closed():
        future = asyncio.run_coroutine_threadsafe(_bot_client.close(), _bot_loop)
        try:
            future.result(timeout=5)
        except Exception:
            pass

    _running = False


def start_worker(process_fn) -> None:
    """Start a background thread that drains the message queue.

    process_fn(discord_msg: DiscordMessage) is called for each message.
    Runs independently of the main REPL thread so Discord messages are
    processed immediately without waiting for the user to press Enter.
    """
    global _worker_thread, _worker_stop
    _worker_stop.clear()

    def _worker():
        while not _worker_stop.is_set():
            try:
                msg = _message_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                process_fn(msg)
            except Exception as e:
                print(f"\n  \033[31m[Discord] Error processing message: {e}\033[0m")

    _worker_thread = threading.Thread(target=_worker, daemon=True, name="discord-worker")
    _worker_thread.start()


def stop_worker() -> None:
    """Stop the Discord worker thread."""
    global _worker_stop, _worker_thread
    _worker_stop.set()
    if _worker_thread:
        _worker_thread.join(timeout=2)
        _worker_thread = None


def send_to_channel(channel_id: int, text: str) -> None:
    """Send a message to a Discord channel from outside the async loop."""
    if not _running or _bot_client is None or _bot_loop is None:
        return

    async def _send():
        channel = _bot_client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await _bot_client.fetch_channel(channel_id)
            except Exception:
                return
        for part in [text[i:i+1900] for i in range(0, len(text), 1900)]:
            await channel.send(part)

    asyncio.run_coroutine_threadsafe(_send(), _bot_loop)


def send_to_last_active(text: str) -> bool:
    """Send text to the most recently active Discord channel. Returns True if attempted."""
    if not _running or _bot_client is None or _bot_loop is None or _last_active_channel is None:
        return False
    channel = _last_active_channel

    async def _send():
        for part in [text[i:i + 1900] for i in range(0, len(text), 1900)]:
            await channel.send(part)

    asyncio.run_coroutine_threadsafe(_send(), _bot_loop)
    return True
