"""``python -m regact.prompt_preview`` — see exactly what the agent will be told.

Prints the two things a run sends: the SYSTEM prompt (framework brief + the game's
section + the always-on controller + any optional feature's fragment + the control and
lifecycle blocks) and the FIRST MESSAGE, which carries the first observation rendered as
the agent will see it.

Useful when tuning prompts or adding a game/feature: it assembles them through the
real :class:`PromptBuilder`, so what you read is what the agent gets — no run, no LLM.

    python -m regact.prompt_preview                          # arc_agi, text render
    python -m regact.prompt_preview --problem minigrid
    python -m regact.prompt_preview --obs raw                # the nested frame instead
    python -m regact.prompt_preview --info-mode minimal      # the discover-it-yourself brief
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from regact.agent.capabilities import TOOL_PROTOCOLS
from regact.config.schema import ControllerConfig, InfoMode, Lifecycle, ObsMode
from regact.env.wrapped_env import WrappedEnv
from regact.features.base import build_features
from regact.features.controller import Controller
from regact.problems.base import BaseProblem, build_problem
from regact.prompt.builder import PromptBuilder

_RAW_PREVIEW_CHARS = 1500


def _configured_lifecycle(problem_name: str) -> Lifecycle:
    """The lifecycle from the problem's own config group, so nothing is hardcoded here."""
    conf = Path(__file__).parent / "conf" / "problem" / f"{problem_name}.yaml"
    try:
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(conf))
        if isinstance(data, dict):
            return Lifecycle(str(data.get("lifecycle", Lifecycle.MULTI_INSTANCE)))
    except (OSError, ValueError, ImportError):
        pass
    return Lifecycle.MULTI_INSTANCE


def _first_obs(problem: BaseProblem, task: str, mode: ObsMode = ObsMode.RAW) -> Any:
    """The observation a fresh env starts on (no action taken)."""
    env = WrappedEnv(
        problem.make_env(task),
        task_name=task,
        renderer=problem.obs_renderer(task, mode=mode),
    )
    return env.last_obs if env.last_obs is not None else env.reset()


def _render_obs(problem: BaseProblem, obs: Any, mode: str) -> str:
    if mode == "raw":
        text = str(obs.frame)
        return text[:_RAW_PREVIEW_CHARS] + (
            " …(truncated)" if len(text) > _RAW_PREVIEW_CHARS else ""
        )
    return problem.render_obs_text(obs) or "(this game renders no text observation)"


def show(
    problem_name: str,
    *,
    obs_mode: str,
    info_mode: InfoMode,
    features: list[str],
    tool_protocol: str,
    fully_obs: bool,
    task: str | None,
) -> None:
    kwargs = {"fully_obs": fully_obs} if problem_name == "minigrid" else {}
    problem = build_problem(problem_name, kwargs)
    task_name = task or problem.get_task_names()[0]
    lifecycle = _configured_lifecycle(problem_name)
    builder = PromptBuilder()

    bar = "=" * 90
    print(f"\n{bar}\n{problem_name}  task={task_name}  lifecycle={lifecycle.value}  obs={obs_mode}")
    print(f"features={features}  tool_protocol={tool_protocol}  info_mode={info_mode.value}\n{bar}")

    obs_mode_enum = ObsMode(obs_mode) if obs_mode in {m.value for m in ObsMode} else ObsMode.RAW
    print("\n##### SYSTEM PROMPT #####\n")
    print(
        builder.build_system_prompt(
            problem,
            task_name,
            build_features({name: {} for name in features}),
            controller=Controller.from_config(ControllerConfig()),
            lifecycle=lifecycle,
            info_mode=info_mode,
            obs_mode=obs_mode_enum,
            tool_protocol=tool_protocol,  # type: ignore[arg-type]
            tool_names=["SubmitSolution", "ExitTask"],
        )
    )

    print("\n##### FIRST MESSAGE #####\n")
    try:
        rendered = _render_obs(problem, _first_obs(problem, task_name, obs_mode_enum), obs_mode)
    except Exception as exc:
        rendered = f"(could not build a live observation: {type(exc).__name__}: {exc})"
    print(builder.build_first_message(rendered))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="regact.prompt_preview", description="Print the prompt a run would send."
    )
    parser.add_argument("--problem", default="arc_agi")
    parser.add_argument("--task", default=None, help="default: the problem's first task")
    parser.add_argument("--obs", default="text", choices=["text", "raw"])
    parser.add_argument("--info-mode", default="informative", choices=[m.value for m in InfoMode])
    parser.add_argument(
        "--feature",
        action="append",
        help="optional feature to add (repeatable); the controller is always-on core",
    )
    parser.add_argument(
        "--tool-protocol",
        default="bash_block",
        choices=list(TOOL_PROTOCOLS),
        help="how the agent invokes tools (default: bash_block, the swegrid fenced-block style)",
    )
    parser.add_argument(
        "--fully-obs",
        action="store_true",
        help="minigrid: full-observation prompt (default off = egocentric partial view)",
    )
    args = parser.parse_args(argv)

    logging.disable(logging.INFO)  # silence the game backend so the prompt reads clean
    show(
        args.problem,
        obs_mode=args.obs,
        info_mode=InfoMode(args.info_mode),
        features=args.feature or [],
        tool_protocol=args.tool_protocol,
        fully_obs=args.fully_obs,
        task=args.task,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
