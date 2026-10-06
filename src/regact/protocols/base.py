"""Experiment protocols: one workflow per task, using the shared runtime.

This is distinct from an agent's ToolProtocol (the syntax used to invoke tools).
A protocol owns its task artifacts, instructions, tools, completion criteria and
reminders. The runner owns transport, sandbox plumbing, event dispatch and limits.
Protocol instances are created per task; mutable state must never be shared
between concurrent tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from regact.agent.capabilities import ToolProtocol
from regact.config.schema import Lifecycle
from regact.envclient.client import EnvClient
from regact.features.base import FeatureContext, Hook
from regact.obs.logger import RunLogger
from regact.orchestration.signals import StopSignal
from regact.problems.base import BaseProblem
from regact.session.state import ExperimentState
from regact.tools.base import Tool
from regact.workspace.templates import TemplateFile


@dataclass(kw_only=True)
class ProtocolContext:
    """Trusted run resources available when binding the protocol's tools.

    No solution or submission paths: those belong to policy_search. A protocol
    chooses its own artifacts under workdir/output_dir and its own worker model.
    sandbox_wrap is the existing evaluation wrapper; isolated CWM workers will
    need their own restricted profile, not the agent's writable workspace.
    """

    experiment: ExperimentState
    env_client: EnvClient
    lifecycle: Lifecycle
    workdir: str
    output_dir: str
    logger: RunLogger
    is_perfect: Callable[[dict[str, Any]], bool]
    # (a submission's results) -> the problem's main metrics for it
    main_metrics: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    failure_metrics: Callable[..., dict[str, Any]] | None = None
    compute_episode_metrics: Callable[..., dict[str, Any]] | None = None
    aggregate_episode_metrics: Callable[..., dict[str, Any]] | None = None
    sandbox_wrap: Callable[[list[str]], list[str]] | None = None
    render_frame: Callable[..., Any] | None = None
    seed: int | None = None


@dataclass(kw_only=True)
class ProtocolSession(ABC):
    """A bound workflow. The common loop does not interpret submission/phase state.

    Completion is polled before turns and after outstanding tools have returned.
    Hooks run at the existing teardown boundary, before after_teardown(). Budget,
    interrupt, backend-error and tool-completion handling remain in the loop.
    """

    tools: list[Tool] = field(default_factory=list)
    hooks: list[Hook] = field(default_factory=list)

    @abstractmethod
    def stop_reason(self) -> str | None:
        """Protocol outcome, or None to continue; must not perform work or mutate state."""
        ...

    @abstractmethod
    def reminder(self, reminders: int) -> str:
        """Instruction for the next ordinary agent turn (not backend-error retries)."""
        ...

    def interaction_guidance(self) -> str:
        return "Discover the game only by playing it through framework/make_env."

    async def prepare(self, stop: StopSignal | None = None) -> None:
        """Optional trusted initialization before the agent starts (never in dry runs)."""
        return None

    def on_start(self, start: float, seconds: float | None) -> None:
        """Receive the common session clock and the task's time budget from it (None = no
        limit): after bootstrap, and again whenever the clock was paused."""
        return None

    async def close(self) -> None:
        """Release protocol resources on every exit, including dry runs."""
        return None

    def resume_notice(self) -> str:
        """Where the task stands, added to the message a resumed agent receives."""
        return ""

    def task_metrics(self) -> dict[str, Any] | None:
        """The problem's main metrics for the task so far, or None when nothing is scored yet."""
        return None

    def after_teardown(self, reason: str, logger: RunLogger) -> None:
        """Optional protocol diagnostics, after finalization and before session_end."""
        return None


class ExperimentProtocol(ABC):
    """Per-task workflow definition, selected once by config and bound once by run_task."""

    name: str
    exposes_environment: bool = True
    resumable: bool = False  # can continue an interrupted task (config.resume)
    command_script: str = "framework/control.py"

    async def __aenter__(self) -> ExperimentProtocol:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release resources acquired before bind, including bootstrap failures."""
        return None

    @abstractmethod
    def validate(self) -> None:
        """Reject unsupported configurations before creating run artifacts."""
        ...

    @abstractmethod
    def templates(self, ctx: FeatureContext) -> Iterable[TemplateFile]:
        """Protocol files, written after the common workspace and problem helpers."""
        ...

    def env_wrappers(self, ctx: FeatureContext) -> list[Callable[[Any], Any]]:
        """Trusted environment wrappers; do not rely on agent-side checks for permissions."""
        return []

    def configure_environment(
        self, server: Any, session: Any, ctx: FeatureContext, problem: BaseProblem
    ) -> None:
        """Optionally own environment permissions and protocol data routes."""
        return None

    @abstractmethod
    def build_system_prompt(
        self,
        problem: BaseProblem,
        task_name: str,
        *,
        tool_protocol: ToolProtocol,
        tool_names: list[str],
        verbalize_variant: str,
    ) -> str:
        """Assemble instructions for this workflow and the selected agent tool syntax."""
        ...

    @abstractmethod
    def bind(self, ctx: ProtocolContext) -> ProtocolSession:
        """Wire tools, finalization and loop behavior to the live task resources."""
        ...
