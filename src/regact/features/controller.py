"""The always-on controller: the core capability the agent always has.

Not a feature - it is fully part of the framework and every run has it. Owns
everything controller-specific: it scaffolds ``base_controller.py``, an example
controller, and the ``solution.py`` stub; explains the controller contract in
the prompt; provides the ``SubmitSolution`` and ``ExitTask`` tools (wired with
the run's executor + experiment); and ships a teardown hook that re-scores the
final ``solution.py`` as the official result. ``make_env.py`` is NOT here - it is
env/lifecycle-specific, written by the workspace base. It reuses the generic
``FeatureContext``/``RunDeps``/``Hook`` seams (shared orchestration types that
live in ``features.base``), but is instantiated directly by the orchestrator from
``config.controller``, never via the feature registry.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from regact.config.schema import ControllerConfig
from regact.controllers.executor import ControllerExecutor, SandboxedExecutor, write_result
from regact.features.base import (
    FeatureContext,
    Hook,
    HookPhase,
    RunDeps,
    TemplateFile,
)
from regact.obs.errors import ErrorCategory
from regact.obs.result import EvalResult
from regact.tools.base import Tool
from regact.tools.exit_task import ExitTask
from regact.tools.submit_solution import SubmitSolution

logger = logging.getLogger(__name__)

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

# A runnable template tying make_env + a controller together, so the agent has one
# obvious file to copy for its own probing scripts. Prints small, readable facts about
# each obs (never the whole frame, which would flood the output).
_INTERACTIVE_SCRIPT = '''\
"""Example interactive script: run a controller on the environment for one episode.

Copy and adapt this to probe the game. It shows the whole loop: make the env, run a
controller, read each observation. It prints only small facts about each obs (never the
full frame - that floods your output); do your real analysis on obs.frame in code.
"""

from framework.make_env import make_env
from code_library.example_controller import ExampleController

MAX_STEPS = 20  # cap the demo to a short run


def main() -> None:
    env = make_env()
    controller = ExampleController()
    obs = env.current()  # the current observation, without spending an action
    steps = 0
    while not obs.is_done and steps < MAX_STEPS:
        action = controller.act(obs)
        obs = env.step(action)
        steps += 1
        # Any observation analysis operation here: inspect obs.frame (a list of 64x64
        # grids) in code - locate objects, diff consecutive frames, count cells, ...
        print(
            f"step={steps} action={action} reward={obs.reward} is_done={obs.is_done} "
            f"n_available={len(obs.available_actions)} available={obs.available_actions} "
            f"n_frames={len(obs.frame)} state={obs.info.get('state')}"
        )
    print(f"episode stopped after {steps} steps: reward={obs.reward} info={obs.info}")


if __name__ == "__main__":
    main()
'''

# The deliverable the agent edits and submits.
_SOLUTION_STUB = '''\
"""Your controller: edit ``act`` and submit this file.

It subclasses ``ExampleController`` (which picks the first available action), so it
runs and scores as-is. Instantiation is the per-episode reset; keep state on
``self``. Never import the game; use ``framework.make_env`` in your own scripts to
test, then submit.
"""

from code_library.example_controller import ExampleController


class Controller(ExampleController):
    def act(self, obs):
        return super().act(obs)  # your policy goes here


def get_controller() -> Controller:
    return Controller()
'''

# The controller feature's prompt fragment lives in markdown next to this module
# (like the system + game prompts), so prose is edited without touching code.
_PROMPT_MD = Path(__file__).parent / "prompts" / "controller.md"


def _make_executor(deps: RunDeps, *, shadow_replay: bool) -> ControllerExecutor | SandboxedExecutor:
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
            shadow_replay=shadow_replay,
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

    def __init__(
        self,
        deps: RunDeps,
        *,
        n_episodes: int,
        max_moves: int,
        n_videos: int,
        shadow_replay: bool,
    ) -> None:
        self._deps = deps
        self._n_episodes = n_episodes
        self._max_moves = max_moves
        self._n_videos = n_videos
        self._shadow_replay = shadow_replay

    async def run(self) -> EvalResult | None:
        deps = self._deps
        if not os.path.exists(deps.solution_path):
            return None  # nothing was ever written; nothing to finalize
        try:
            # Offload the blocking eval to a thread, exactly as SubmitSolution does: the
            # netbridge loopback relay runs on THIS event loop, so a synchronous
            # subprocess.run here would starve it and the sandboxed eval's env calls would
            # time out (ReadTimeout) instead of scoring.
            executor = _make_executor(deps, shadow_replay=self._shadow_replay)
            result = await asyncio.to_thread(
                executor.run,
                task_name=deps.experiment.task_name,
                solution_path=deps.solution_path,
                output_path=os.path.join(deps.submissions_dir, "final", "results.json"),
                lifecycle=deps.lifecycle,
                n_episodes=self._n_episodes,
                max_moves=self._max_moves,
                n_videos=self._n_videos,
            )
        except Exception as exc:
            # The out-of-process eval can raise at teardown past its own catches (e.g. a parent-side
            # read timeout); log the traceback and record it gracefully so finalize never aborts.
            logger.exception("finalize re-score crashed: %s", deps.experiment.task_name)
            result = EvalResult(
                task=deps.experiment.task_name,
                error=f"{type(exc).__name__}: {exc}",
                error_category=ErrorCategory.EVAL_HARNESS,
                executor="finalize",
            )
        if deps.feature_metrics is not None:
            result.features = deps.feature_metrics()
            write_result(os.path.join(deps.submissions_dir, "final", "results.json"), result)
        deps.experiment.last_submission_results = result.to_json()
        return result


class Controller:
    """The always-on controller-writing capability (core, not a feature).

    Owns its evaluation knobs (they configure the controller's scoring): ``n_episodes``
    eval episodes per submission, ``max_moves`` env steps per rollout, ``n_videos``
    eval episodes to record a video of, ``shadow_replay`` for the anti-cheat re-score. Built by the
    orchestrator from :class:`ControllerConfig`; exposes the same ``templates`` /
    ``prompt_fragment`` / ``tools`` / ``hooks`` seams a feature does, applied alongside
    the optional features.
    """

    evaluates_on_env = True  # SubmitSolution/finalize score by rolling episodes on the env

    def __init__(
        self,
        *,
        n_episodes: int = 1,
        max_moves: int = 2500,
        n_videos: int = 2,
        shadow_replay: bool = False,
    ) -> None:
        self._n_episodes = int(n_episodes)
        self._max_moves = int(max_moves)
        self._n_videos = int(n_videos)
        self._shadow_replay = bool(shadow_replay)

    @classmethod
    def from_config(cls, config: ControllerConfig) -> Controller:
        """Build the controller from its run-config section."""
        return cls(
            n_episodes=config.n_episodes,
            max_moves=config.max_moves,
            n_videos=config.n_videos,
            shadow_replay=config.shadow_replay,
        )

    def templates(self, ctx: FeatureContext) -> list[TemplateFile]:
        return [
            TemplateFile("code_library/base_controller.py", _BASE_CONTROLLER),
            TemplateFile("code_library/example_controller.py", _EXAMPLE_CONTROLLER),
            TemplateFile("code_library/interactive_script_example.py", _INTERACTIVE_SCRIPT),
            TemplateFile("solution.py", _SOLUTION_STUB),
        ]

    def prompt_fragment(self, ctx: FeatureContext) -> str | None:
        return _PROMPT_MD.read_text(encoding="utf-8")

    def tools(self, deps: RunDeps) -> list[Tool]:
        # This feature owns the eval, so it also owns the eval fields of the state.
        deps.experiment.n_eval_episodes = self._n_episodes
        deps.experiment.n_videos = min(self._n_videos, self._n_episodes)
        submit = SubmitSolution(
            deps.experiment,
            _make_executor(deps, shadow_replay=self._shadow_replay),
            solution_path=deps.solution_path,
            submissions_dir=deps.submissions_dir,
            task_name=deps.experiment.task_name,
            lifecycle=deps.lifecycle,
            n_episodes=self._n_episodes,
            max_moves=self._max_moves,
            n_videos=self._n_videos,
            feature_metrics=deps.feature_metrics,
        )
        return [submit, ExitTask(deps.experiment)]

    def hooks(self, deps: RunDeps) -> list[Hook]:
        # ShadowReplayHook (anti-cheat) joins this list in Block 10.
        return [
            FinalizeControllerHook(
                deps,
                n_episodes=self._n_episodes,
                max_moves=self._max_moves,
                n_videos=self._n_videos,
                shadow_replay=self._shadow_replay,
            )
        ]
