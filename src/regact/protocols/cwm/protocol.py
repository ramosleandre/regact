"""Phase-based code world-model learning, independent of policy-search evaluation."""

from collections.abc import Iterator

from regact.agent.capabilities import ToolProtocol, uses_control_cli
from regact.features.base import FeatureContext
from regact.problems.base import BaseProblem
from regact.protocols.cwm.commands import enabled_commands
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.prompting import build_prompt
from regact.protocols.cwm.session import Coordinator
from regact.protocols.cwm.templates import templates
from regact.protocols.managed.protocol import ManagedProtocol
from regact.workspace.templates import TemplateFile


class CwmProtocol(ManagedProtocol):
    name = "cwm"
    coordinator_type = Coordinator
    config_type = CwmConfig

    def templates(self, ctx: FeatureContext) -> Iterator[TemplateFile]:
        commands = self.coordinator.commands if self.coordinator else enabled_commands(self.options)
        yield from templates(ctx, self.options, commands=commands, vision=self.config.agent.vision)

    def build_system_prompt(
        self,
        problem: BaseProblem,
        task_name: str,
        *,
        tool_protocol: ToolProtocol,
        tool_names: list[str],
        verbalize_variant: str,
    ) -> str:
        if self.coordinator is not None:
            self.coordinator.commands_cli = uses_control_cli(tool_protocol)
        return build_prompt(
            problem,
            task_name,
            self.config,
            self.options,
            tool_protocol=tool_protocol,
            tool_names=tool_names,
            verbalize_variant=verbalize_variant,
            command_script=self.command_script,
        )
