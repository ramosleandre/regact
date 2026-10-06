"""Claude Code CLI adapter.

Spawns ``claude -p ... --output-format stream-json`` headless in the workdir and
maps its stream-json events to the normalized union. Auth defaults to the CLI's
own login (subscription); we never pass an API key unless one is configured.
Resume across turns uses the session id Claude reports in its ``init`` event.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from regact.agent.base import executable_paths
from regact.agent.capabilities import Capabilities
from regact.agent.cli_agent import _CliAgent
from regact.agent.events import (
    AgentError,
    AgentEvent,
    IterationComplete,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    tool_result_images,
)
from regact.agent.usage import claude_usage
from regact.obs.errors import ErrorCategory
from regact.security.policy import SecurityPolicy, default_policy

_IMAGE_SUFFIXES = ("png", "jpg", "jpeg", "gif", "webp")
_CLAUDE_BASH_MAX_TIMEOUT_MS = 600_000  # Claude Code's own default ceiling for a Bash timeout


def claude_deny_settings(
    workdir: str, policy: SecurityPolicy | None = None, *, deny_images: bool = False
) -> dict[str, Any]:
    """Claude-native defense-in-depth: deny Claude's file tools from reading game data.

    Backend-specific (Claude's ``.claude/settings.json``), so it lives with the adapter,
    like codex's ``--sandbox`` flags and Alan's PreToolUse hook live with theirs; the
    generic ``security/`` layer stays backend-agnostic. It governs only Claude's native
    Read tool, never arbitrary code the agent runs, so it is defense-in-depth on top of
    the OS sandbox, not a substitute for it.
    """
    policy = policy or default_policy()
    deny = [f"Read(**/{sub.rstrip('/')}/**)" for sub in sorted(policy.forbidden_path_substrings)]
    if deny_images:  # a text-only model rejects any request that carries an image
        deny += [f"Read(**/*.{suffix})" for suffix in _IMAGE_SUFFIXES]
    return {"permissions": {"deny": deny}}


# "You've hit your session limit - resets 7:20pm (Europe/Paris)"; weekly limits add a date
# ("resets Oct 12, 8am (Europe/Paris)").
_LIMIT_RESET = re.compile(
    r"limit.*resets\s+(?:(?P<month>[A-Za-z]{3})\s+(?P<day>\d{1,2}),?\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<half>am|pm)\s*\((?P<zone>[^)]+)\)",
    re.IGNORECASE,
)


_RESET_JUST_PASSED = timedelta(minutes=30)  # an error seen this soon after its own reset time


def limit_reset_unix(message: str, now: float) -> float | None:
    """When the usage limit in a Claude Code error message resets, as UNIX time: the next
    occurrence of the stated local time after ``now``, or that time today when it has only just
    passed (the error arrived as the limit reset). ``None`` for any other message."""
    match = _LIMIT_RESET.search(message)
    if match is None:
        return None
    try:
        zone = ZoneInfo(match["zone"].strip())
        current = datetime.fromtimestamp(now, zone)
        hour = int(match["hour"]) % 12 + (12 if match["half"].lower() == "pm" else 0)
        minute = int(match["minute"] or 0)
        reset = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if match["month"]:
            dated = datetime.strptime(f"{match['month']} {match['day']}", "%b %d")
            reset = reset.replace(month=dated.month, day=dated.day)
            if reset <= current:
                reset = reset.replace(year=reset.year + 1)
        elif reset <= current - _RESET_JUST_PASSED:
            reset += timedelta(days=1)
    except (ValueError, KeyError):
        return None
    return reset.timestamp()


class ClaudeAgent(_CliAgent):
    """``CodeAgent`` backed by the headless Claude Code CLI."""

    def __init__(self, args: dict[str, object] | None = None, *, vision: bool = False) -> None:
        super().__init__(args, vision=vision)
        raw_home = str(self._args.get("claude_home") or "~/.regact/claude-home")
        self._home_root = os.path.realpath(os.path.expanduser(raw_home))
        self._session_home: str | None = None  # this task's fresh config dir (created on demand)
        self._usage: dict[str, Any] | None = None

    def _real_creds(self) -> str:
        return os.path.join(os.path.expanduser("~"), ".claude", ".credentials.json")

    def _can_isolate(self) -> bool:
        """Relocate to an isolated dir only when auth survives it: a copyable ``.credentials.json``
        exists (file-based login) or the caller forced a home. Else Keychain-only macOS auth, which
        is keyed to the DEFAULT dir, would be lost and the CLI would strand as "Not logged in"."""
        forced = self._args.get("claude_home") is not None
        return (
            forced
            or os.path.exists(os.path.join(self._home_root, ".credentials.json"))
            or os.path.exists(self._real_creds())
        )

    def _freshest_creds(self) -> str | None:
        """The NEWEST existing credential of {isolated root, real ~/.claude} - seed the LIVE token,
        never a stale copy. OAuth rotates the refresh token, so a stale copy reads back as
        'revoked'. Handles both logins: into ~/.claude (its copy is newer) or into the root."""
        candidates = [os.path.join(self._home_root, ".credentials.json"), self._real_creds()]
        existing = [c for c in candidates if os.path.exists(c)]
        return max(existing, key=os.path.getmtime) if existing else None

    def _make_session_home(self) -> str:
        """A FRESH per-task config dir seeded with ONLY the (freshest) auth credential - no
        projects/memory, sessions, or history from any prior task or run (which a shared home would
        accumulate). The root persists the login; each task gets its own empty dir under it."""
        home = os.path.join(self._home_root, "session", uuid.uuid4().hex)
        os.makedirs(home, exist_ok=True)
        src = self._freshest_creds()
        if src is not None:
            shutil.copyfile(src, os.path.join(home, ".credentials.json"))
        return home

    def _config_dir(self) -> str:
        """The config dir claude will actually use — decided independently of start(), and cached so
        host_rw_paths()/auth_check()/_configure_home() all agree. A fresh per-task home when we can
        isolate without losing auth; else the real ``~/.claude`` (Keychain-only macOS auth)."""
        if not self._can_isolate():
            return os.path.join(os.path.expanduser("~"), ".claude")
        if self._session_home is None:
            self._session_home = self._make_session_home()
        return self._session_home

    def _configure_workdir(self) -> None:
        # Native confinement: a .claude/settings.json deny-list keeps Claude's file
        # tools inside the workdir (it cannot read the game data outside it).
        settings_dir = os.path.join(self._cwd, ".claude")
        os.makedirs(settings_dir, exist_ok=True)
        with open(os.path.join(settings_dir, "settings.json"), "w", encoding="utf-8") as handle:
            json.dump(
                claude_deny_settings(self._cwd, deny_images=not self._vision), handle, indent=2
            )
        self._configure_home()
        if self._base_url:  # an Anthropic-compatible server, e.g. llama.cpp's llama-server
            self._env_overrides |= {
                "ANTHROPIC_BASE_URL": self._base_url,
                "ANTHROPIC_AUTH_TOKEN": self._api_key or "local",
                "ANTHROPIC_API_KEY": "",
                "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            }
            if self._args.get("context_window"):
                window = str(int(self._args["context_window"]))  # sizes its compaction
                self._env_overrides["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = window
        budget = self._args.get("max_thinking_tokens")
        if budget:
            self._env_overrides["MAX_THINKING_TOKENS"] = str(budget)
        timeout = self._args.get("bash_timeout_ms")
        if timeout:  # the Bash timeout when the model passes none; its own cap must not undercut it
            ms = int(timeout)
            self._env_overrides["BASH_DEFAULT_TIMEOUT_MS"] = str(ms)
            self._env_overrides["BASH_MAX_TIMEOUT_MS"] = str(max(ms, _CLAUDE_BASH_MAX_TIMEOUT_MS))

    def _configure_home(self) -> None:
        """Point claude at its config dir. When we isolate, that dir is a fresh per-task home seeded
        with only auth (see :meth:`_make_session_home`), so no prior session's memory / transcript /
        history leaks in. On Keychain-only auth we leave ``CLAUDE_CONFIG_DIR`` unset so claude keeps
        its real home + auth."""
        config_dir = self._config_dir()
        if config_dir == os.path.join(os.path.expanduser("~"), ".claude"):
            return  # Keychain-only auth: real home, relocating would drop auth
        self._env_overrides["CLAUDE_CONFIG_DIR"] = config_dir

    async def close(self) -> None:
        """Drop the per-task config home on teardown (nothing reads claude's native session dir
        post-run; the normalized transcript is already in logs/), so seeded auth + session state do
        not accumulate. First preserve any token refresh Claude wrote back to the isolated ROOT
        (never the user's ~/.claude) - dropping a rotated refresh token revokes the persistent one.
        """
        await super().close()
        if self._session_home is None:
            return
        refreshed = os.path.join(self._session_home, ".credentials.json")
        if os.path.exists(refreshed):
            try:
                os.makedirs(self._home_root, exist_ok=True)
                shutil.copyfile(refreshed, os.path.join(self._home_root, ".credentials.json"))
            except OSError:
                pass  # best-effort; a lost refresh just re-seeds from ~/.claude next run
        self._usage = claude_usage(self._session_home)  # before the home and its logs are deleted
        shutil.rmtree(self._session_home, ignore_errors=True)
        self._session_home = None

    def usage_limit_reset(self, message: str) -> float | None:
        return limit_reset_unix(message, time.time())

    def usage(self) -> dict[str, Any] | None:
        return self._usage

    def prompt_for_transcript(self, prepared: str) -> str:
        return "[Claude Code system prompt — supplied by Claude Code, not captured]\n\n" + prepared

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="append",  # --append-system-prompt
            tool_protocol="client_cli",  # native bash/file tools; submit/exit via the workdir CLI
            permission_hooks=True,  # .claude/settings.json deny-list + permission mode
            streams_tool_calls=True,
            supports_inject=False,  # per-turn resume; injection is prepended next turn
            writes_native_transcript=True,  # .claude session dir
        )

    def launch_probe_argv(self) -> list[str]:
        """Cheap liveness check: the Claude CLI must be executable inside the sandbox."""
        return ["claude", "--version"]

    def auth_check(self) -> tuple[str, str] | None:
        """Detect the "Not logged in" case without spending a real turn.

        ``claude -p`` with a trivial prompt errors immediately with an auth message when
        unauthenticated (or when the config dir was relocated away from Keychain auth),
        so we can catch it cheaply. Uses the same ``CLAUDE_CONFIG_DIR`` a real run would.
        """
        if shutil.which("claude") is None:
            return "warn", "'claude' not on PATH"
        env = dict(os.environ)
        config_dir = self._config_dir()
        if config_dir != os.path.join(os.path.expanduser("~"), ".claude"):
            env["CLAUDE_CONFIG_DIR"] = config_dir
        try:
            proc = subprocess.run(
                ["claude", "-p", "hi", "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return "warn", f"could not run auth check ({type(exc).__name__})"
        out = (proc.stdout or "") + (proc.stderr or "")
        if "Not logged in" in out or "authentication_failed" in out or "Please run /login" in out:
            return "warn", "not logged in — run `claude` once to authenticate"
        if proc.returncode != 0 and "rate" in out.lower():
            return "warn", "authenticated but rate-limited (out of credits / 5h window)"
        return "ok", "authenticated"

    def host_read_paths(self) -> list[str]:
        home = os.path.expanduser("~")
        paths = [
            *executable_paths("claude"),  # the CLI's bin dir + its real install tree
            os.path.join(home, ".npm"),  # package cache (npm installs); no session data
            os.path.join(home, ".claude.json"),
        ]
        if sys.platform == "darwin":
            claude_tmp = f"/tmp/claude-{os.getuid()}"
            os.makedirs(claude_tmp, exist_ok=True)  # must exist => a (subpath) rule, not (literal)
            paths += [os.path.join(home, "Library/Keychains"), "/Library/Keychains", claude_tmp]
        return paths

    def host_rw_paths(self) -> list[str]:
        home = self._config_dir()  # the dir claude truly writes to (isolated or real ~/.claude)
        os.makedirs(home, exist_ok=True)  # must exist for a bind/subpath rule
        return [home]

    def host_egress_hosts(self) -> list[str]:
        return self._egress_hosts(["api.anthropic.com"])  # not statsig / sentry telemetry

    def host_write_prefixes(self) -> list[str]:
        if sys.platform != "darwin":
            return []
        return [os.path.realpath("/tmp") + "/claude-"]

    def _command(self, message: str) -> tuple[list[str], str | None]:
        argv = ["claude", "-p", message, "--output-format", "stream-json", "--verbose"]
        argv += ["--permission-mode", str(self._args.get("permission_mode", "bypassPermissions"))]
        if self._args.get("effort"):
            argv += ["--effort", str(self._args["effort"])]
        if self._session_id is not None:
            argv += ["--resume", self._session_id]
        elif self._system_prompt:
            argv += ["--append-system-prompt", self._system_prompt]
        if self._model:
            argv += ["--model", self._model]
        return argv, None  # message is passed as the -p argument, not stdin

    def _track_session(self, obj: dict[str, Any]) -> None:
        session_id = obj.get("session_id")
        if isinstance(session_id, str):
            self._session_id = session_id

    def _parse_events(self, obj: dict[str, Any]) -> list[AgentEvent]:
        kind = obj.get("type")
        if kind == "assistant":
            return _blocks_to_events(_content(obj))
        if kind == "user":
            return [
                ToolResult(
                    id=str(block.get("tool_use_id", "")),
                    output=_text_of(block.get("content")),
                    is_error=bool(block.get("is_error", False)),
                    images=tool_result_images(block.get("content")),
                )
                for block in _content(obj)
                if block.get("type") == "tool_result"
            ]
        if kind == "result":
            if obj.get("is_error") or obj.get("subtype") not in (None, "success"):
                return [AgentError(ErrorCategory.AGENT_API, _text_of(obj.get("result")))]
            usage = obj.get("usage")
            return [
                IterationComplete(
                    final_text=_text_of(obj.get("result")),
                    usage=usage if isinstance(usage, dict) else None,
                )
            ]
        return []  # "system"/init and anything else: tracked or ignored


def _content(obj: dict[str, Any]) -> list[dict[str, Any]]:
    message = obj.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []


def _blocks_to_events(blocks: list[dict[str, Any]]) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            events.append(TextDelta(_text_of(block.get("text"))))
        elif btype == "thinking":
            text = _text_of(block.get("thinking"))
            if text:
                events.append(ThinkingDelta(text))
        elif btype == "tool_use":
            tool_input = block.get("input")
            events.append(
                ToolCall(
                    id=str(block.get("id", "")),
                    name=str(block.get("name", "")),
                    input=tool_input if isinstance(tool_input, dict) else {},
                )
            )
    return events


def _text_of(value: Any) -> str:
    """Claude content can be a string or a list of text blocks; flatten to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(b.get("text", "") for b in value if isinstance(b, dict))
    return "" if value is None else str(value)
