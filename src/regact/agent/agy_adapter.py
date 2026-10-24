"""Antigravity CLI (``agy``) adapter.

Spawns ``agy -p`` headless in the workdir and maps its ``stream-json`` events to the normalized
union; resumes across turns by the conversation id agy reports. Event mapping follows agy 1.4.7:
an ``init`` event (conversation id), ``step_update`` events (agent text deltas, tool calls with
their output, per-step token usage) and a final ``result``.

agy runs against a fresh per-task home holding regact's agent definition: its instructions
(regact's system prompt) and its tool allowlist, which leaves out agy's web search, URL reader,
browser, image generation and subagents. Inside the sandbox the home is read-only except the
folders agy writes, so the agent cannot widen its own tools.

Authentication: ``auth: login`` uses the account the user signed in with (agy keeps its token in
the desktop keyring, which the sandbox cannot reach, so login runs only unsandboxed for now);
``auth: api_key`` uses ``GEMINI_API_KEY`` and runs sandboxed.
"""

from __future__ import annotations

import glob
import os
import shutil
import uuid
from collections import defaultdict
from typing import Any

from regact.agent.base import executable_paths
from regact.agent.capabilities import Capabilities
from regact.agent.cli_agent import _CliAgent
from regact.agent.events import (
    AgentError,
    AgentEvent,
    IterationComplete,
    TextDelta,
    ToolCall,
    ToolResult,
)
from regact.obs.errors import ErrorCategory, RegactError

_AGENT_NAME = "regact"
# Shell and file tools; everything else agy offers (web, browser, images, subagents, scheduling,
# workflows, MCP, questions to a user who is not there) is left out.
_TOOLS = (
    "run_command",
    "write_to_file",
    "replace_file_content",
    "multi_replace_file_content",
    "list_dir",
    "grep_search",
    "find_by_name",
)
_IMAGE_TOOLS = ("view_file",)  # it also shows images to the model
_AUTH_MODES = ("login", "api_key")


class AgyAgent(_CliAgent):
    """``CodeAgent`` backed by the headless Antigravity CLI."""

    _env_drop = ("GEMINI_", "GOOGLE_", "ANTIGRAVITY")

    def __init__(self, args: dict[str, object] | None = None, *, vision: bool = False) -> None:
        super().__init__(args, vision=vision)
        self._auth = str(self._args.get("auth") or "login")
        if self._auth not in _AUTH_MODES:
            raise ValueError(f"agent.args.auth must be one of {_AUTH_MODES}, not {self._auth!r}")
        self._env_keep = ("GEMINI_API_KEY",) if self._auth == "api_key" else ()
        raw_home = str(self._args.get("agy_home") or "~/.regact/agy-home")
        self._home_root = os.path.realpath(os.path.expanduser(raw_home))
        self._session_home: str | None = None
        self._totals: dict[str, int] = defaultdict(int)
        self._processes = 0  # agy processes run so far: step indexes restart in each one
        self._text: dict[Any, list[str]] = {}  # an answer's streamed pieces, by step
        self._started: set[str] = set()

    def usage(self) -> dict[str, Any] | None:
        if not any(self._totals.values()):
            return None
        return {"source": "agy_stream", "models": {self._model or "default": dict(self._totals)}}

    def prompt_for_transcript(self, prepared: str) -> str:
        return "[Antigravity system prompt - supplied by agy, not captured]\n\n" + prepared

    def capabilities(self) -> Capabilities:
        return Capabilities(
            system_prompt="append",  # the agent definition's instructions, after agy's own
            tool_protocol="client_cli",  # native shell/file tools; submit/exit via the workdir CLI
            permission_hooks=False,
            streams_tool_calls=True,
            supports_inject=False,
            writes_native_transcript=True,  # conversations in the isolated home
        )

    def launch_probe_argv(self) -> list[str]:
        return ["agy", "--version"]

    def host_read_paths(self) -> list[str]:
        return [*executable_paths("agy"), self._config_dir()]

    def host_rw_paths(self) -> list[str]:
        home = self._config_dir()
        return [
            os.path.join(home, ".gemini", "antigravity-cli"),
            os.path.join(home, ".gemini", "config"),
            os.path.join(home, ".config", "antigravity"),
        ]

    def host_egress_hosts(self) -> list[str]:
        if self._auth == "api_key":
            return ["generativelanguage.googleapis.com"]
        return [
            "daily-cloudcode-pa.googleapis.com",
            "cloudcode-pa.googleapis.com",
            "oauth2.googleapis.com",
            "www.googleapis.com",
        ]

    def _config_dir(self) -> str:
        """A fresh per-task home: regact's agent definition and the folders agy writes."""
        if self._session_home is None:
            home = os.path.join(self._home_root, "session", uuid.uuid4().hex)
            for sub in (("antigravity-cli",), ("config",), ("agents",)):
                os.makedirs(os.path.join(home, ".gemini", *sub), exist_ok=True)
            os.makedirs(os.path.join(home, ".config", "antigravity"), exist_ok=True)
            self._session_home = home
        return self._session_home

    def _configure_workdir(self) -> None:
        if self._auth == "api_key" and not os.environ.get("GEMINI_API_KEY"):
            raise RegactError(
                ErrorCategory.AGENT_API, "agent.args.auth=api_key needs GEMINI_API_KEY"
            )
        home = self._config_dir()
        self._env_overrides["HOME"] = home
        tools = [*_TOOLS, *(_IMAGE_TOOLS if self._vision else ())]
        definition = (
            "---\n"
            f"name: {_AGENT_NAME}\n"
            "description: regact benchmark agent\n"
            f"tools: [{', '.join(tools)}]\n"
            "---\n" + (self._system_prompt or "") + "\n"
        )
        with open(
            os.path.join(home, ".gemini", "agents", f"{_AGENT_NAME}.md"), "w", encoding="utf-8"
        ) as handle:
            handle.write(definition)

    async def close(self) -> None:
        """Keep agy's own transcripts (what the model saw, thinking included) in
        logs/agy_transcripts/, then drop the per-task home."""
        await super().close()
        if self._session_home is None:
            return
        if self._cwd:
            kept = os.path.join(os.path.dirname(self._cwd), "logs", "agy_transcripts")
            brain = os.path.join(self._session_home, ".gemini", "antigravity-cli", "brain")
            for path in glob.glob(os.path.join(brain, "*", ".system_generated", "logs", "*.jsonl")):
                conversation = path.split(os.sep)[-4]
                try:
                    os.makedirs(kept, exist_ok=True)
                    shutil.copyfile(
                        path, os.path.join(kept, f"{conversation}_{os.path.basename(path)}")
                    )
                except OSError:
                    pass  # best-effort: the regact transcript is the record
        if not self.keep_session:
            shutil.rmtree(self._session_home, ignore_errors=True)
            self._session_home = None

    def _command(self, message: str) -> tuple[list[str], str | None]:
        self._processes += 1
        argv = [
            "agy",
            "--agent",
            _AGENT_NAME,
            "-p",
            message,
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
        ]
        if self._model:
            argv += ["--model", self._model]
        if self._args.get("reasoning_effort"):
            argv += ["--effort", str(self._args["reasoning_effort"])]
        if self._session_id is not None:
            argv += ["--conversation", self._session_id]
        return argv, None

    def _track_session(self, obj: dict[str, Any]) -> None:
        conversation = obj.get("conversation_id")
        if obj.get("event") == "init" and isinstance(conversation, str) and conversation:
            self._session_id = conversation

    def _parse_events(self, obj: dict[str, Any]) -> list[AgentEvent]:
        kind = obj.get("event")
        if kind == "result":
            result = obj.get("result") or {}
            if result.get("status") != "SUCCESS":
                error = str(result.get("error") or result)
                if "authentication" in error and self._auth == "login":
                    error += (
                        " (the login token is in the desktop keyring, which the sandbox cannot "
                        "reach: run with sandbox=false, or use agent.args.auth=api_key)"
                    )
                return [AgentError(ErrorCategory.AGENT_API, f"agy: {error}")]
            return [IterationComplete(final_text=str(result.get("response") or ""))]
        step = obj.get("step_update")
        if kind != "step_update" or not isinstance(step, dict):
            return []
        index = step.get("step_index")
        if step.get("step_type") == "agent_response":
            # Each model response reports its tokens: counted here, so a run killed before its
            # final result (a limit) still has its usage.
            for name, value in (step.get("usage") or {}).items():
                if isinstance(value, int) and name.endswith("_tokens") and name != "total_tokens":
                    self._totals[name.removesuffix("_tokens")] += value
            pieces = self._text.setdefault(index, [])
            if step.get("text_delta"):
                pieces.append(str(step["text_delta"]))
            if step.get("state") != "DONE":
                return []
            text = "".join(self._text.pop(index, []))
            return [TextDelta(text)] if text.strip() else []
        if step.get("step_type") != "tool":
            return []
        info = step.get("tool_info") or {}
        tool_id = f"p{self._processes}:{index}"
        events: list[AgentEvent] = []
        if tool_id not in self._started:
            self._started.add(tool_id)
            name = str(step.get("tool_name") or info.get("name") or "tool")
            events.append(ToolCall(tool_id, name, _tool_input(info.get("parameters") or {})))
        if step.get("state") == "DONE":
            failed = "error" in step or step.get("status") == "ERROR"
            events.append(ToolResult(tool_id, str(info.get("output") or ""), failed))
        return events


def _tool_input(parameters: dict[str, Any]) -> dict[str, Any]:
    """agy's arguments, with a shell command and a file path under the names the other agents
    use. agy's stream leaves out written file contents; its own transcript keeps them."""
    out = dict(parameters)
    for native, common in (("CommandLine", "command"), ("TargetFile", "file_path")):
        if native in out:
            out[common] = out.pop(native)
    return out
