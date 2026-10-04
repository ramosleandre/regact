"""Small, documented agent workspace. Enforcement remains in the trusted coordinator."""

from collections.abc import Iterator

from regact.features.base import FeatureContext
from regact.protocols.cwm.commands import enabled_commands
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.workspace_helpers import CWM_ENV, SIMULATE
from regact.protocols.managed.templates import templates as common_templates
from regact.workspace.templates import TemplateFile


def templates(
    ctx: FeatureContext, options: CwmConfig | None = None, *, commands=None, vision: bool = False
) -> Iterator[TemplateFile]:
    options = options or CwmConfig()
    commands = commands or enabled_commands(options)
    yield from common_templates(ctx, options, commands, vision=vision)
    if options.workspace_helpers_enabled:
        yield TemplateFile("framework/cwm_env.py", CWM_ENV)
        yield TemplateFile("simulate.py", SIMULATE)
    from regact.protocols.cwm.prompting import workspace_docs

    yield from workspace_docs(options, vision=vision)
    yield TemplateFile("world_model/__init__.py", "")
    yield TemplateFile(
        "world_model/model_state.py",
        '''"""Define a compact state that preserves all information needed to render and predict."""
from dataclasses import dataclass

@dataclass(frozen=True)
class State:
    """Replace this stub with your learned fields; serialized field names also cost bytes."""
    pass
''',
    )
    for name, signature, doc in [
        (
            "parser",
            "parse(obs)",
            "Build the State from the FIRST observation of a chain (the run start, each multi-instance RunController, a reset without a reset hook). Later States come from step. Do not mutate obs.",
        ),
        (
            "render",
            "render(state)",
            "Reconstruct the complete observation dictionary: frame, reward, is_done, available_actions and info. Do not mutate state.",
        ),
        (
            "transition",
            "step(state, action)",
            "Predict the State after one problem-format action, without real interaction. Carry everything later observations depend on, including what the screen does not show. Do not mutate either input. Optional: define reset(state, kind) for explicit resets (kind is level or environment) to keep hidden state through them; without it the reset observation is parsed.",
        ),
    ]:
        yield TemplateFile(
            f"world_model/model_{name}.py",
            f'from world_model.model_state import State\n\ndef {signature}:\n    """{doc}"""\n    raise NotImplementedError("Implement using recorded experience")\n',
        )
    if "PlanInCWM" in commands:
        yield TemplateFile(
            "goal.py",
            '''"""Replace this docstring with the experimental goal (used for logging)."""

def achieved(state):
    """Return bool: has this predicted state achieved your chosen goal? Must be repeatable."""
    raise NotImplementedError("Define the goal in the CWM state space")

# Optional: utility(state) -> float in [0, 1], with 1 whenever achieved(state).
# Without it, the planner uses float(achieved(state)). Do not mutate state.
''',
        )
    yield TemplateFile(
        "controller.py",
        '''"""Replace this docstring with the experimental goal (used for logging)."""

class ExplorationController:
    """Acts on the State carried by the accepted CWM; each real exploration gets a fresh instance."""
    def act(self, state):
        """Return one action in the problem's format. Private memory is allowed."""
        raise NotImplementedError("Choose an action for this state")

    def is_done(self, state):
        """Return True to stop this exploration early; framework limits still apply."""
        return False

    # Optional: objective_reached(self, state) -> bool for a distinct goal-achieved signal.


def get_controller():
    """Return a new controller. Do not read the data API inside submitted callbacks."""
    return ExplorationController()
''',
    )
