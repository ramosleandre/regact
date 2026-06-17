"""The minimal, always-on feature: writing a controller.

Owns everything controller-specific: it scaffolds ``base_controller.py``, an
example controller, and the ``solution.py`` stub; explains the controller
contract in the prompt; provides the ``SubmitSolution`` and ``ExitTask`` tools
(wired with the run's executor + experiment); and ships a shadow-replay eval hook.
``make_env.py`` is NOT here — it is env/lifecycle-specific, written by the
workspace base.
"""

from __future__ import annotations

from regact.features.base import (
    EvalHook,
    Feature,
    FeatureContext,
    RunDeps,
    TemplateFile,
    register_feature,
)
from regact.obs.result import EvalResult
from regact.tools.base import Tool
from regact.tools.exit_task import ExitTask
from regact.tools.submit_solution import SubmitSolution

# Copied verbatim into the workdir as ``code_library/base_controller.py`` — the
# contract the agent inherits. Mirrors ``regact.controllers.base.BaseController``.
_BASE_CONTROLLER = '''\
"""The controller contract. Subclass this and implement ``act``."""


class BaseController:
    def __init__(self, *args, **kwargs):
        # Set up for a fresh episode; keep cross-step state on self.
        pass

    def act(self, obs):
        # Return an action the env accepts for the given observation.
        raise NotImplementedError
'''

# A worked example the agent can read and adapt.
_EXAMPLE_CONTROLLER = '''\
"""Example controller: always picks the first available action."""

from code_library.base_controller import BaseController


class ExampleController(BaseController):
    def act(self, obs):
        return obs.available_actions[0]
'''

# The deliverable the agent edits and submits.
_SOLUTION_STUB = '''\
"""Your controller. Implement ``act`` and submit this file.

Instantiation is the per-episode reset; keep state on ``self``. Never import the
game — use ``framework.make_env`` in your own scripts to test, then submit.
"""

from code_library.base_controller import BaseController


class Controller(BaseController):
    def __init__(self):
        super().__init__()
        # Reset any per-episode state here.

    def act(self, obs):
        # Return an action drawn from obs.available_actions.
        raise NotImplementedError


def get_controller() -> Controller:
    return Controller()
'''

_PROMPT_FRAGMENT = """\
## Your deliverable: a controller

A controller is a pure policy: `act(obs) -> action`. Instantiation is the
per-episode reset (state lives on `self`); it never receives or imports the env.

Edit `solution.py`:

```python
from code_library.base_controller import BaseController

class Controller(BaseController):
    def act(self, obs):
        return ...  # an action from obs.available_actions

def get_controller() -> Controller:
    return Controller()
```

`code_library/base_controller.py` holds the contract; `example_controller.py`
shows a trivial policy. When ready, call **SubmitSolution** to score it, iterate,
and call **ExitTask** when you are done.
"""


class ControllerEvalHook(EvalHook):
    """Shadow-replay: re-run the kept controller on a fresh, differently-seeded env.

    A behavioural anti-cheat check (a seed-hardcoded policy fails it). The real
    replay logic lands with the anti-cheat harness (Block 10); for now it is a
    no-op that reports nothing flagged so the seam exists and the loop can call it.
    """

    def verify(self, *, workdir: str, session: object) -> EvalResult:
        return EvalResult(task="shadow_replay", aggregate={"flagged": False})


class ControllerFeature(Feature):
    """The base controller-writing capability."""

    name = "controller"

    def templates(self, ctx: FeatureContext) -> list[TemplateFile]:
        return [
            TemplateFile("code_library/base_controller.py", _BASE_CONTROLLER),
            TemplateFile("code_library/example_controller.py", _EXAMPLE_CONTROLLER),
            TemplateFile("solution.py", _SOLUTION_STUB),
        ]

    def prompt_fragment(self, ctx: FeatureContext) -> str | None:
        return _PROMPT_FRAGMENT

    def tools(self, deps: RunDeps) -> list[Tool]:
        submit = SubmitSolution(
            deps.experiment,
            deps.executor,
            solution_path=deps.solution_path,
            submissions_dir=deps.submissions_dir,
            task_name=deps.experiment.task_name,
            lifecycle=deps.lifecycle,
            n_episodes=deps.n_episodes,
            max_moves=deps.max_moves,
        )
        return [submit, ExitTask(deps.experiment)]

    def eval_hooks(self, deps: RunDeps) -> list[EvalHook]:
        return [ControllerEvalHook()]


register_feature(ControllerFeature.name, ControllerFeature)
