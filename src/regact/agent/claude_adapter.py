"""Claude Code CLI adapter.

Spawns ``claude -p ... --output-format stream-json`` headless in the workdir and
maps its stream-json events to the normalized union. Auth defaults to the CLI's
own login (subscription); we never pass an API key unless one is configured.
Resume across turns uses the session id Claude reports in its ``init`` event.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from typing import Any

from regact.agent.base import executable_paths
from regact.agent.capabilities import Capabilities
from regact.agent.cli_agent import _CliAgent
from regact.agent.events import (
    AgentError,
    AgentEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)
from regact.obs.errors import ErrorCategory
from regact.security.policy import SecurityPolicy, default_policy


def claude_deny_settings(workdir: str, policy: SecurityPolicy | None = None) -> dict[str, Any]:
    """Claude-native defense-in-depth: deny Claude's file tools from reading game data.

    Backend-specific (Claude's ``.claude/settings.json``), so it lives with the adapter,
    like codex's ``--sandbox`` flags and Alan's PreToolUse hook live with theirs; the
    generic ``security/`` layer stays backend-agnostic. It governs only Claude's native
    Read tool, never arbitrary code the agent runs, so it is defense-in-depth on top of
    the OS sandbox, not a substitute for it.
    """
    policy = policy or default_policy()
    deny = [f"Read(**/{sub.rstrip('/')}/**)" for sub in sorted(policy.forbidden_path_substrings)]
    return {"permissions": {"deny": deny}}


class ClaudeAgent(_CliAgent):
    """``CodeAgent`` backed by the headless Claude Code CLI."""

    def __init__(self, args: dict[str, object] | None = None) -> None:
        super().__init__(args)
        raw_home = str(self._args.get("claude_home") or "~/.regact/claude-home")
        self._home_root = os.path.realpath(os.path.expanduser(raw_home))
        self._session_home: str | None = None  # this task's fresh config dir (created on demand)

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
            json.dump(claude_deny_settings(self._cwd), handle, indent=2)
        self._configure_home()
        budget = self._args.get("max_thinking_tokens")
        if budget:
            self._env_overrides["MAX_THINKING_TOKENS"] = str(budget)

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
        shutil.rmtree(self._session_home, ignore_errors=True)
        self._session_home = None

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
        return ["api.anthropic.com"]  # block statsig.anthropic.com / sentry telemetry

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
                )
                for block in _content(obj)
                if block.get("type") == "tool_result"
            ]
        if kind == "result":
            if obj.get("is_error") or obj.get("subtype") not in (None, "success"):
                return [AgentError(ErrorCategory.AGENT_API, _text_of(obj.get("result")))]
            usage = obj.get("usage")
            return [
                TurnComplete(
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
