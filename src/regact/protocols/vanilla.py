"""Observation-driven baseline using the same managed execution as CWM."""

import math
from dataclasses import asdict, dataclass, field

from regact.features.base import FeatureContext
from regact.protocols.cwm.config import DataApiConfig, FeedbackConfig
from regact.protocols.managed.prompting import build_prompt, reset_commands
from regact.protocols.managed.protocol import ManagedProtocol
from regact.protocols.managed.templates import templates as common_templates
from regact.workspace.templates import TemplateFile


@dataclass
class ExecutionConfig:
    max_seconds_per_call: float | None = 5
    max_seconds_per_RunController: float | None = 90
    max_memory_mb: int | None = 512


@dataclass
class VanillaConfig:
    n_unique_observations_in_initial_collection: int = 20
    max_actions_per_initial_collection: int | None = 1000
    max_seconds_per_initial_collection: float | None = 30
    max_actions_per_RunController: int | None = 2500
    n_tmp_images_saved_per_exploration: int = 0
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    data_api: DataApiConfig = field(default_factory=DataApiConfig)

    @classmethod
    def from_mapping(cls, raw):
        values = dict(raw)
        for key, typ in (
            ("execution", ExecutionConfig),
            ("feedback", FeedbackConfig),
            ("data_api", DataApiConfig),
        ):
            values[key] = typ(**dict(values.get(key) or {}))
        result = cls(**values)

        def check(values):
            for key, value in values.items():
                if isinstance(value, dict):
                    check(value)
                elif key.startswith("max_") and value is None:
                    continue
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    or (value == 0 and key != "n_tmp_images_saved_per_exploration")
                ):
                    raise ValueError(f"{key} must be positive (max_* also accepts null)")
                elif "seconds" not in key and type(value) is not int:
                    raise ValueError(f"{key} must be an integer")

        check(asdict(result))
        return result


CONTROLLER = '''"""Describe your intended experiment here."""
class ExplorationController:
    """A fresh instance is created for every RunController call."""
    def act(self, obs):
        """Return one problem-format action from the full observation dictionary."""
        raise NotImplementedError("Choose an action")

    def is_done(self, obs):
        """Return True to stop this call. False leaves stopping to the environment/budgets."""
        return False

def get_controller():
    """Return a fresh controller; submitted code cannot access the dataset or network."""
    return ExplorationController()
'''


class VanillaProtocol(ManagedProtocol):
    name = "vanilla"
    config_type = VanillaConfig

    def commands(self, problem=None):
        if self.coordinator is not None:
            return self.coordinator.commands
        return {
            "RunController": "Run a fresh controller from controller.py.",
            **(reset_commands(self.config, problem) if problem else {}),
        }

    def templates(self, ctx):
        yield from common_templates(
            ctx, self.options, self.commands(), vision=self.config.agent.vision
        )
        yield TemplateFile("controller.py", CONTROLLER)

    def build_system_prompt(
        self, problem, task_name, *, tool_protocol, tool_names, verbalize_variant
    ):
        ctx = FeatureContext(problem_name=problem.name, task_name=task_name, workdir="")
        return build_prompt(
            problem,
            task_name,
            self.config,
            self.options,
            files=[
                *common_templates(
                    ctx, self.options, self.commands(problem), vision=self.config.agent.vision
                ),
                TemplateFile("controller.py", CONTROLLER),
            ],
            tool_protocol=tool_protocol,
            tool_names=tool_names,
            verbalize_variant=verbalize_variant,
            command_script=self.command_script,
        )
