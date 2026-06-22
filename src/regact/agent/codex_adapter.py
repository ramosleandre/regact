"""Codex CLI adapter.

Spawns ``codex exec --json`` headless in the workdir (prompt piped on stdin) and
maps its ndjson events to the normalized union; resumes across turns via the
thread id codex reports. Ported in shape from arc-3-agents-baseline1's
``codex_runner`` (subprocess + ndjson + thread-id resume).

NOTE: codex's exact ``--json`` event schema is not pinned here, so ``_parse_events``
is best-effort (keyed on common fields) and should be validated against real codex
output — the live test is gated on the ``codex`` CLI being installed.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

from regact.agent.capabilities import Capabilities
from regact.agent.cli_agent import _CliAgent
from regact.agent.events import (
    AgentEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)


class CodexAgent(_CliAgent):
    """``CodeAgent`` backed by the headless codex CLI."""

    def __init__(self, args: dict[str, object] | None = None) -> None:
        super().__init__(args)
        # Run codex against a generated, isolated home rather than the user's ~/.codex, so
        # the session is reproducible and carries no ambient user config. Kept outside the
        # per-run workdir so the auth token stays out of run artifacts. Override via
        # ``agent.args["codex_home"]``.
        raw_home = str(self._args.get("codex_home") or "~/.regact/codex-home")
        self._codex_home = os.path.realpath(os.path.expanduser(raw_home))

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="replace",
            control_actions="client_cli",
            permission_hooks=False,
            streams_tool_calls=True,
            supports_inject=False,
            writes_native_transcript=True,  # session store in the isolated home
        )

    def host_read_paths(self) -> list[str]:
        # The isolated home holds codex's config, auth, and session store (read-write).
        return [self._codex_home]

    def host_egress_hosts(self) -> list[str]:
        # API-key mode needs only api.openai.com; ChatGPT-login adds auth/chatgpt.
        return ["api.openai.com", "auth.openai.com", "chatgpt.com"]

    def _configure_workdir(self) -> None:
        """Seed the isolated home and point codex at it via ``CODEX_HOME`` and ``HOME``.

        Codex reads config and skills from both ``$CODEX_HOME`` and ``$HOME/.agents``;
        redirecting both at the empty home yields a clean session (no ambient config,
        plugins, or MCP servers). Writes a minimal ``config.toml`` and copies the auth
        token if present (auth is file-based under ``CODEX_HOME``); otherwise codex falls
        back to ``OPENAI_API_KEY``.
        """
        home = self._codex_home
        os.makedirs(os.path.join(home, "skills"), exist_ok=True)
        with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as handle:
            handle.write("# generated: isolated codex home\n")
        user_auth = os.path.join(os.path.expanduser("~"), ".codex", "auth.json")
        if os.path.exists(user_auth):
            shutil.copyfile(user_auth, os.path.join(home, "auth.json"))
        self._env_overrides["CODEX_HOME"] = home
        self._env_overrides["HOME"] = home

    def _command(self, message: str) -> tuple[list[str], str | None]:
        argv = ["codex"]
        if self._model:
            argv += ["-m", self._model]
        if self._args.get("reasoning_effort"):
            argv += ["-c", f"model_reasoning_effort={self._args['reasoning_effort']}"]
        # Default: bypass approvals + sandbox so the agent can reach the localhost
        # env/control server (codex's own sandbox would block it); our path scan +
        # the HTTP boundary are the confinement. Override via agent.args.sandbox /
        # agent.args.ask_for_approval if you want codex's native sandbox instead.
        if self._args.get("sandbox"):
            argv += ["--sandbox", str(self._args["sandbox"])]
            argv += ["--ask-for-approval", str(self._args.get("ask_for_approval", "never"))]
        else:
            argv += ["--dangerously-bypass-approvals-and-sandbox"]
        argv += ["exec", "--cd", os.path.abspath(self._cwd) if self._cwd else ".", "--json"]
        if self._session_id is not None:
            argv += ["resume", self._session_id]
        return argv, message  # codex reads the prompt from stdin

    def _track_session(self, obj: dict[str, Any]) -> None:
        thread_id = obj.get("thread_id")
        thread = obj.get("thread")
        if thread_id is None and isinstance(thread, dict):
            thread_id = thread.get("id")
        if isinstance(thread_id, str):
            self._session_id = thread_id

    def _parse_events(self, obj: dict[str, Any]) -> list[AgentEvent]:
        """Map one codex ndjson object to events.

        Codex streams an item lifecycle: ``item.started`` (a command begins) then
        ``item.completed`` (it finished, with output). We emit a clean ``ToolCall`` on
        start and a ``ToolResult`` on completion (paired by id), so the transcript reads
        like the native-tool agents instead of duplicating the raw item dict.
        """
        kind = str(obj.get("type", ""))
        raw = obj.get("item")
        item: dict[str, Any] = raw if isinstance(raw, dict) else obj
        itype = item.get("type")

        if "reasoning" in kind or itype == "reasoning":
            return [ThinkingDelta(_text_of(item.get("text") or item.get("reasoning")))]

        if itype in ("command_execution", "tool_call", "function_call", "mcp_tool_call"):
            tool_id = str(item.get("id", ""))
            if kind.endswith("item.completed") or item.get("status") == "completed":
                exit_code = item.get("exit_code")
                return [
                    ToolResult(
                        id=tool_id,
                        output=_text_of(item.get("aggregated_output") or item.get("output")),
                        is_error=isinstance(exit_code, int) and exit_code != 0,
                    )
                ]
            if kind.endswith("item.started"):
                return [ToolCall(id=tool_id, name=_tool_name(item), input=_tool_input(item))]
            return []  # item.updated and other intermediate frames

        if kind.endswith("turn.completed") or kind.endswith("turn_complete"):
            return [TurnComplete(final_text=_text_of(item.get("text")))]
        message = item.get("message")
        text = item.get("text") or (message if isinstance(message, str) else None)
        if text:
            return [TextDelta(_text_of(text))]
        return []


def _tool_name(item: dict[str, Any]) -> str:
    """A short tool label: ``shell`` for a command, else the tool/function name."""
    if item.get("command") is not None:
        return "shell"
    return str(item.get("name") or item.get("tool") or item.get("type") or "tool")


def _tool_input(item: dict[str, Any]) -> dict[str, Any]:
    """The tool's arguments only — never the noisy lifecycle/output fields."""
    if item.get("command") is not None:
        return {"command": item["command"]}
    drop = {"id", "type", "status", "aggregated_output", "exit_code"}
    return {k: v for k, v in item.items() if k not in drop}


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(b.get("text", "") for b in value if isinstance(b, dict))
    return "" if value is None else str(value)
