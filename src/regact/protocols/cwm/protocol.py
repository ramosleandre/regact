"""Phase-based code world-model learning, independent of policy-search evaluation."""

from collections.abc import Iterator
from pathlib import Path

from regact.agent.capabilities import ToolProtocol, uses_control_cli
from regact.config.schema import Lifecycle, RunConfig
from regact.env.server import EnvServer
from regact.env.session import EnvSession
from regact.envclient.obs import Obs
from regact.features.base import FeatureContext
from regact.problems.base import BaseProblem
from regact.prompt.builder import _terminal_block
from regact.protocols.base import ExperimentProtocol, ProtocolContext
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.session import Coordinator, CwmSession, CwmTool
from regact.protocols.cwm.templates import templates
from regact.security.runtime import SandboxRuntime, resolve
from regact.workspace.templates import TemplateFile

NAMES = ("UpdateCodeWorldModel", "PlanInCWM", "SubmitExplorationController")


class CwmProtocol(ExperimentProtocol):
    name = "cwm"

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
            self.config.limits.max_real_actions_per_task is not None
            and self.config.limits.max_real_actions_per_task <= 0
        ):
            raise ValueError("limits.max_real_actions_per_task must be positive or null")

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
        yield from templates(ctx)

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
        commands = (
            "\n".join(f"python framework/control.py {name}" for name in tool_names)
            if uses_control_cli(tool_protocol)
            else ", ".join(tool_names)
        )
        return "\n\n".join(
            filter(
                None,
                [
                    (
                        "# Learn a code world model and use it to solve the environment\n"
                        "Write and test Python code in your workspace. "
                        "Read CWM_INTERFACE.md for the exact interfaces."
                    ),
                    _terminal_block(tool_protocol).replace("solution.py", "exploration.py"),
                    problem.build_prompt(
                        task_name,
                        info_mode=self.config.problem.info_mode,
                        obs_mode=self.config.problem.obs_mode,
                    ),
                    f"""# Workflow
Phase 0: interact through `from framework.make_env import make_env`, then env.step(action).
Collect {self.options.initial_unique_observations} unique complete observations.
The initial observation counts.
The action reaching this threshold returns normally; later direct environment calls are blocked.
Phase 1: use framework.cwm_data to inspect real experience and implement world_model/.
Call UpdateCodeWorldModel. Acceptance requires exact reconstruction and prediction on ALL recorded
evidence, distinct states for distinct observations, and aggregate state/observation byte ratio
strictly below {self.options.threshold_state_obs_size_ratio}.
Phase 2: choose and describe a useful experimental goal, write exploration.py, and call
SubmitExplorationController. Optionally write goal.py and call PlanInCWM first.
A novel predicted observation is required for real execution. Novelty is evidence for testing
your model, not a score to maximize. Investigate mechanisms and progress toward solving the task.
The framework dream-checks your controller, then runs a fresh instance
from the fixed real start.
Dream and real execution have separate action allowances:
{self.options.max_actions_per_exploration} actions each.
The first predictive/reconstruction mismatch stops real execution and returns to phase 1.
Finishing an experiment without contradiction returns to phase 2; this is a normal outcome.
The run ends on full game completion or configured limits. Keep working until then.

Submitted code runs isolated from the experience database, workdir, game engine and network.
Use the data API while developing; callbacks must use only their explicit inputs and bundled code.
Do not try to access the live environment from a model or controller callback.

# Framework commands
{commands}""",
                ],
            )
        )
