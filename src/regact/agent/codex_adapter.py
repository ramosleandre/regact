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
from typing import Any

from regact.agent.capabilities import Capabilities
from regact.agent.cli_agent import _CliAgent
from regact.agent.events import AgentEvent, TextDelta, ThinkingDelta, ToolCall, TurnComplete


class CodexAgent(_CliAgent):
    """``CodeAgent`` backed by the headless codex CLI."""

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="replace",
            control_actions="client_cli",
            permission_hooks=False,
            streams_tool_calls=True,
            supports_inject=False,
            writes_native_transcript=True,  # ~/.codex session
        )

    def host_read_paths(self) -> list[str]:
        # ~/.codex holds codex's config + auth + the session sqlite (read-write).
        return [os.path.join(os.path.expanduser("~"), ".codex")]

    def host_egress_hosts(self) -> list[str]:
        # API-key mode needs only api.openai.com; ChatGPT-login adds auth/chatgpt.
        return ["api.openai.com", "auth.openai.com", "chatgpt.com"]

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
        argv += ["exec"]
        if self._session_id is not None:
            argv += ["resume", self._session_id]
        argv += ["--cd", self._cwd or ".", "--json"]
        return argv, message  # codex reads the prompt from stdin

    def _track_session(self, obj: dict[str, Any]) -> None:
        thread_id = obj.get("thread_id")
        thread = obj.get("thread")
        if thread_id is None and isinstance(thread, dict):
            thread_id = thread.get("id")
        if isinstance(thread_id, str):
            self._session_id = thread_id

    def _parse_events(self, obj: dict[str, Any]) -> list[AgentEvent]:
        kind = str(obj.get("type", ""))
        raw = obj.get("item")
        item: dict[str, Any] = raw if isinstance(raw, dict) else obj
        itype = item.get("type")

        if "reasoning" in kind or itype == "reasoning":
            return [ThinkingDelta(_text_of(item.get("text") or item.get("reasoning")))]
        if itype in ("command_execution", "tool_call", "function_call"):
            return [
                ToolCall(
                    id=str(item.get("id", "")),
                    name=str(item.get("name") or item.get("command") or "command"),
                    input=item,
                )
            ]
        if kind.endswith("turn.completed") or kind.endswith("turn_complete"):
            return [TurnComplete(final_text=_text_of(item.get("text")))]
        message = item.get("message")
        text = item.get("text") or (message if isinstance(message, str) else None)
        if text:
            return [TextDelta(_text_of(text))]
        return []


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(b.get("text", "") for b in value if isinstance(b, dict))
    return "" if value is None else str(value)
