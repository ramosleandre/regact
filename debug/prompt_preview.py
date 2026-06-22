"""Render the full prompt for a game: system prompt + first message with a rendered obs.

Run:  make debug D=prompt_preview                          # arc_agi, grid render
      make debug D=prompt_preview ARGS="arc_agi grid"      # compact 64x64 hex grid
      make debug D=prompt_preview ARGS="arc_agi raw"       # raw obs.frame (truncated)
      make debug D=prompt_preview ARGS="minigrid grid"

Shows the system prompt and the first message, which embeds the first observation
rendered the chosen way (`grid` = compact hex, `raw` = the nested frame). The observation
comes from the made env (no action). Falls back to a note if the backend is not installed.
"""

import logging
import sys
from typing import Any

logging.disable(logging.INFO)  # silence the game backend's INFO chatter for a clean view

from regact.config.schema import InfoMode, Lifecycle, ObsMode
from regact.env.wrapped_env import WrappedEnv
from regact.features.controller import ControllerFeature
from regact.problems.base import BaseProblem, build_problem
from regact.prompt.builder import PromptBuilder

_TOOLS = ["SubmitSolution", "ExitTask"]
_LIFECYCLE = {"arc_agi": Lifecycle.SINGLE_INSTANCE, "minigrid": Lifecycle.MULTI_INSTANCE}


def _first_obs(problem: BaseProblem, task: str) -> Any:
    native = problem.make_env(task)
    renderer = problem.obs_renderer(task, mode=ObsMode.RAW)
    env = WrappedEnv(native, task_name=task, renderer=renderer)
    return env.last_obs if env.last_obs is not None else env.reset()  # init obs, else reset


def _render_obs(problem: BaseProblem, obs: Any, mode: str) -> str:
    if mode == "raw":
        text = str(obs.frame)
        return text[:1500] + (" …(truncated)" if len(text) > 1500 else "")
    return problem.render_obs_text(obs) or "(this game has no text obs render yet)"


def show(problem_name: str, render_mode: str) -> None:
    problem = build_problem(problem_name, {})
    task = problem.get_task_names()[0]
    builder = PromptBuilder()
    features = [ControllerFeature()]
    lifecycle = _LIFECYCLE.get(problem_name, Lifecycle.MULTI_INSTANCE)

    bar = "=" * 90
    print(f"\n{bar}\n{problem_name}  task={task}  lifecycle={lifecycle.value}  "
          f"obs_render={render_mode}\n{bar}")
    print("\n##### SYSTEM PROMPT (informative) #####\n")
    print(
        builder.build_system_prompt(
            problem, task, features,
            lifecycle=lifecycle, info_mode=InfoMode.INFORMATIVE,
            control_actions="client_cli", tool_names=_TOOLS,
        )
    )

    print("\n##### FIRST MESSAGE #####\n")
    try:
        rendered = _render_obs(problem, _first_obs(problem, task), render_mode)
    except Exception as exc:  # noqa: BLE001 — preview tool, surface backend errors
        rendered = f"(could not build a live observation: {type(exc).__name__}: {exc})"
    print(builder.build_first_message(rendered))


def main() -> None:
    args = sys.argv[1:]
    show(args[0] if args else "arc_agi", args[1] if len(args) > 1 else "grid")


if __name__ == "__main__":
    main()
