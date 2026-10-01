"""Phase-based code world-model learning, independent of policy-search evaluation."""

from pathlib import Path

from regact.config.schema import RunConfig
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.obs import Obs
from regact.features.base import FeatureContext
from regact.problems.base import BaseProblem
from regact.protocols.base import ExperimentProtocol, ProtocolContext
from regact.protocols.managed.session import ManagedCoordinator, ManagedSession, ManagedTool
from regact.security.runtime import SandboxRuntime, resolve


class ManagedProtocol(ExperimentProtocol):
    """Common environment binding, worker requirements and tool lifetime."""

    exposes_environment = False
    command_script = "framework/commands.py"
    coordinator_type = ManagedCoordinator

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.options = self.config_type.from_mapping(config.protocol.options)
        self.coordinator = None

    def validate(self) -> None:
        if self.config.features:
            raise ValueError(
                "Managed controller protocols do not compose legacy policy-search features; use features=none"
            )
        if (
            resolve(SandboxRuntime(self.config.sandbox_opts.get("backend", "auto")))
            is SandboxRuntime.NONE
        ):
            raise ValueError(
                "Managed controller protocols require OS-isolated workers, even when the agent sandbox is disabled"
            )
        if (
            self.config.limits.max_actions_per_task is not None
            and self.config.limits.max_actions_per_task <= 0
        ):
            raise ValueError("limits.max_actions_per_task must be positive or null")

    def configure_environment(
        self, server: EnvServer, session: EnvSession, ctx: FeatureContext, problem: BaseProblem
    ) -> None:
        c = self.coordinator_type(
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

    def bind(self, ctx: ProtocolContext) -> ManagedSession:
        assert self.coordinator is not None
        self.coordinator.context = ctx
        return ManagedSession(
            coordinator=self.coordinator,
            tools=[ManagedTool(name, self.coordinator) for name in self.coordinator.commands],
        )
