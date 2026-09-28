"""Phase-based code world-model learning, independent of policy-search evaluation."""

from collections.abc import Iterator
from pathlib import Path

from regact.agent.capabilities import ToolProtocol
from regact.config.schema import Lifecycle, RunConfig
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.obs import Obs
from regact.features.base import FeatureContext
from regact.problems.base import BaseProblem
from regact.protocols.base import ExperimentProtocol, ProtocolContext
from regact.protocols.cwm.commands import COMMANDS
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.prompting import build_prompt
from regact.protocols.cwm.session import Coordinator, CwmSession, CwmTool
from regact.protocols.cwm.templates import templates
from regact.security.runtime import SandboxRuntime, resolve
from regact.workspace.templates import TemplateFile

NAMES = tuple(COMMANDS)


class CwmProtocol(ExperimentProtocol):
    name = "cwm"
    exposes_environment = False

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.options = CwmConfig.from_mapping(config.protocol.options)
        self.coordinator: Coordinator | None = None

    def validate(self) -> None:
        if self.config.problem.lifecycle is not Lifecycle.MULTI_INSTANCE:
            raise ValueError("CWM v4 currently requires problem.lifecycle=multi_instance")
        if self.config.features:
            raise ValueError(
                "protocol=cwm does not compose legacy policy-search features; use features=none"
            )
        if (
            resolve(SandboxRuntime(self.config.sandbox_opts.get("backend", "auto")))
            is SandboxRuntime.NONE
        ):
            raise ValueError(
                "CWM requires OS-isolated workers, even when the agent sandbox is disabled"
            )
        if (
            self.config.limits.max_actions_per_task is not None
            and self.config.limits.max_actions_per_task <= 0
        ):
            raise ValueError("limits.max_actions_per_task must be positive or null")

    def configure_environment(
        self, server: EnvServer, session: EnvSession, ctx: FeatureContext, problem: BaseProblem
    ) -> None:
        c = Coordinator(
            self.config,
            self.options,
            session,
            problem,
            ctx.task_name,
            Path(ctx.output_dir),
            Path(ctx.workdir),
        )
        self.coordinator = c
        server.bind_environment(
            ctx.task_name, c.public_environment, lambda: Obs.from_json(c.initial), c.data
        )

    async def close(self) -> None:
        if self.coordinator is not None:
            context = self.coordinator.context
            await self.coordinator.shutdown(context.experiment.exit_reason if context else None)

    def templates(self, ctx: FeatureContext) -> Iterator[TemplateFile]:
        yield from templates(ctx, self.options)

    def bind(self, ctx: ProtocolContext) -> CwmSession:
        assert self.coordinator is not None
        self.coordinator.context = ctx
        return CwmSession(
            coordinator=self.coordinator, tools=[CwmTool(name, self.coordinator) for name in NAMES]
        )

    def build_system_prompt(
        self,
        problem: BaseProblem,
        task_name: str,
        *,
        tool_protocol: ToolProtocol,
        tool_names: list[str],
        verbalize_variant: str,
    ) -> str:
        return build_prompt(
            problem,
            task_name,
            self.config,
            self.options,
            tool_protocol=tool_protocol,
            tool_names=tool_names,
            verbalize_variant=verbalize_variant,
        )
