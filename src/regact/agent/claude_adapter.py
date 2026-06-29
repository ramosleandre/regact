"""Claude Code CLI adapter.

Spawns ``claude -p ... --output-format stream-json`` headless in the workdir and
maps its stream-json events to the normalized union. Auth defaults to the CLI's
own login (subscription); we never pass an API key unless one is configured.
Resume across turns uses the session id Claude reports in its ``init`` event.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

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

    def _configure_workdir(self) -> None:
        # Native confinement: a .claude/settings.json deny-list keeps Claude's file
        # tools inside the workdir (it cannot read the game data outside it).
        settings_dir = os.path.join(self._cwd, ".claude")
        os.makedirs(settings_dir, exist_ok=True)
        with open(os.path.join(settings_dir, "settings.json"), "w", encoding="utf-8") as handle:
            json.dump(claude_deny_settings(self._cwd), handle, indent=2)
        budget = self._args.get("max_thinking_tokens")
        if budget:
            self._env_overrides["MAX_THINKING_TOKENS"] = str(budget)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="append",  # --append-system-prompt
            control_actions="client_cli",  # reaches submit/exit via the workdir CLI
            permission_hooks=True,  # .claude/settings.json deny-list + permission mode
            streams_tool_calls=True,
            supports_inject=False,  # per-turn resume; injection is prepended next turn
            writes_native_transcript=True,  # .claude session dir
        )

    def host_read_paths(self) -> list[str]:
        home = os.path.expanduser("~")
        paths = [
            os.path.join(home, ".claude"),
            os.path.join(home, ".claude.json"),
            os.path.join(home, ".npm"),
        ]
        if sys.platform == "darwin":
            claude_tmp = f"/tmp/claude-{os.getuid()}"
            os.makedirs(claude_tmp, exist_ok=True)  # must exist => a (subpath) rule, not (literal)
            paths += [os.path.join(home, "Library/Keychains"), "/Library/Keychains", claude_tmp]
        return paths

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
