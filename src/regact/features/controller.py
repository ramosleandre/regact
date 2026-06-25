"""The minimal, always-on feature: writing a controller.

Owns everything controller-specific: it scaffolds ``base_controller.py``, an
example controller, and the ``solution.py`` stub; explains the controller
contract in the prompt; provides the ``SubmitSolution`` and ``ExitTask`` tools
(wired with the run's executor + experiment); and ships a teardown hook that
re-scores the final ``solution.py`` as the official result.
``make_env.py`` is NOT here — it is env/lifecycle-specific, written by the
workspace base.
"""

from __future__ import annotations

import os
from pathlib import Path

from regact.controllers.executor import ControllerExecutor, SandboxedExecutor
from regact.features.base import (
    Feature,
    FeatureContext,
    Hook,
    HookPhase,
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

# The controller feature's prompt fragment lives in markdown next to this module
# (like the system + game prompts), so prose is edited without touching code.
_PROMPT_MD = Path(__file__).parent / "prompts" / "controller.md"


def _make_executor(deps: RunDeps) -> ControllerExecutor | SandboxedExecutor:
    """Pick how the controller is evaluated: a sandboxed subprocess for real runs (a
    real HTTP env to dial), in-process for the ``scripted`` test backend (no socket)."""
    if deps.sandbox_wrap is not None:
        return SandboxedExecutor(
            workdir=os.path.dirname(deps.solution_path),
            sandbox_wrap=deps.sandbox_wrap,
            compute_metrics=deps.compute_episode_metrics,
            aggregate_metrics=deps.aggregate_episode_metrics,
            render_frame=deps.render_frame,
            seed=deps.seed,
            env_client=deps.env_client,
            shadow_replay=deps.shadow_replay,
        )
    return ControllerExecutor(
        deps.env_client,
        compute_metrics=deps.compute_episode_metrics,
        aggregate_metrics=deps.aggregate_episode_metrics,
        render_frame=deps.render_frame,
        seed=deps.seed,
    )


class FinalizeControllerHook(Hook):
    """Teardown: re-score the *final* ``solution.py`` as the official result.

    Guards against the common failure where the agent edits ``solution.py`` and
    exits (or runs out of turns) without re-submitting, so the last numbered
    submission no longer reflects the file on disk. The loop fires this on every
    non-aborted exit path; the result is written to ``submissions/final``.
    """

    phase = HookPhase.TEARDOWN

    def __init__(self, deps: RunDeps) -> None:
        self._deps = deps

    async def run(self) -> EvalResult | None:
        deps = self._deps
        if not os.path.exists(deps.solution_path):
            return None  # nothing was ever written; nothing to finalize
        result = _make_executor(deps).run(
            task_name=deps.experiment.task_name,
            solution_path=deps.solution_path,
            output_path=os.path.join(deps.submissions_dir, "final", "results.json"),
            lifecycle=deps.lifecycle,
            n_episodes=deps.n_episodes,
            max_moves=deps.max_moves,
            record_video=deps.record_video,
        )
        deps.experiment.last_submission_results = result.to_json()
        return result


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
        return _PROMPT_MD.read_text(encoding="utf-8")

    def tools(self, deps: RunDeps) -> list[Tool]:
        submit = SubmitSolution(
            deps.experiment,
            _make_executor(deps),
            solution_path=deps.solution_path,
            submissions_dir=deps.submissions_dir,
            task_name=deps.experiment.task_name,
            lifecycle=deps.lifecycle,
            n_episodes=deps.n_episodes,
            max_moves=deps.max_moves,
            record_video=deps.record_video,
        )
        return [submit, ExitTask(deps.experiment)]

    def hooks(self, deps: RunDeps) -> list[Hook]:
        # ShadowReplayHook (anti-cheat) joins this list in Block 10.
        return [FinalizeControllerHook(deps)]


register_feature(ControllerFeature.name, ControllerFeature)
