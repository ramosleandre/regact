"""CWM-specific content; shared PromptBuilder owns agent dialect and assembly."""

from collections.abc import Iterator
from pathlib import Path

from regact.agent.capabilities import ToolProtocol
from regact.config.schema import RunConfig
from regact.features.base import FeatureContext
from regact.problems.base import BaseProblem
from regact.protocols.cwm.commands import enabled_commands
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.limits import describe
from regact.protocols.cwm.unknown import MAX_UNKNOWN_SHARE, UNKNOWN, UNKNOWN_FRAMES
from regact.protocols.managed.prompting import action_cap_sentence, reset_commands
from regact.protocols.managed.prompting import build_prompt as build_managed_prompt
from regact.workspace.templates import TemplateFile

_PROMPTS = Path(__file__).with_name("prompts")
_DOCS = {
    "modeling_phase.md": "docs/modeling_phase.md",
    "acting_phase.md": "docs/acting_phase.md",
    "plan_in_CWM.md": "docs/plan_in_CWM.md",
}
_DESCRIPTIONS = {
    "docs/modeling_phase.md": "how to implement, validate and repair the CWM",
    "docs/acting_phase.md": "how to act in the environment with an accepted CWM",
    "docs/plan_in_CWM.md": "(optional) how to use the CWM-dependent planner tool",
    "framework/cwm_env.py": "EnvCWM, to run your CWM like an environment",
    "goal.py": "the file to write goals before using the PlanInCWM command",
    "world_model/model_initial_state.py": (
        'define your "get_initial_state: observation (dictionary) -> State"'
    ),
    "world_model/model_render.py": 'define your "render: State -> observation (dictionary)"',
    "world_model/model_state.py": 'define your class "State"',
    "world_model/model_transition.py": 'define your "step: State, action -> State"',
    "world_model/model_explain.py": '(optional) define your "explain: State, action, obs -> choices"',
    "world_model/model_reset.py": '(optional) define your "reset: State, obs -> State"',
    "framework/cwm_stochastic.py": "run and score your stochastic CWM locally",
}
_LOCAL_HELPERS = """
## Local simulation helpers

`framework/cwm_env.py` provides `EnvCWM` and `make_cwm_env`. Read their docstrings for arguments and examples. They use the current workspace CWM; they do not validate or submit it, update the dataset, or perform real actions.

`EnvCWM(initial_state=state)` requires an explicit State and never queries the dataset. `make_cwm_env()` instead starts where the next controller starts: in single-instance mode it rebuilds the current State from the live episode with your workspace CWM (`get_initial_state` on its first observation, then `step` through every recorded action); in multi-instance mode it uses `get_initial_state` on the original initial observation. Passing initial_state skips that lookup. Both return States from reset()/step(action), or full observation dictionaries with obs_mode=True. Reset restores the starting State without taking a real action.

`serialized_size(value)` returns the bytes the compactness check counts for a State or an observation, and `serialize(value)` the JSON text it measures.

Local execution uses the shell tool's timeout, not isolated-callback limits. Start a new script after edits. Validate changed CWM files with UpdateCodeWorldModel before running a controller. Submitted callbacks must supply a State explicitly, not load data through the factory.
"""

_STOCHASTIC_INTRO = """
- **A stochastic model.** This environment is random or hides part of its state. Your `step` names its random choices, a recorded step must be *explained* by values of those choices, and each explanation costs bits. Read "Stochastic world model" at the end of `docs/modeling_phase.md` before writing `step`.
"""

_STOCHASTIC_ACTING = """
**With a stochastic CWM, read this document with one change.** The CWM does not predict an observation before the action. After each real action, the framework *explains* the new observation: it looks for values of your `step`'s choices under which `render` gives exactly that observation, and the State they produce is carried on. Where this document says "prediction", read "the observation given by the most probable values of your choices". A step with no explanation is the contradiction described below. A step explained by a very unlikely value stops the call as a *surprise*, which you can tune from your controller. See "Stochastic world model" in `docs/modeling_phase.md`.
"""

_STOCHASTIC_HELPERS = """
## Local helpers

`framework/cwm_stochastic.py` runs your current workspace CWM locally; it submits nothing and takes no real action. `current_state()` returns the State your CWM gives for the live episode. `explain_step(state, action, obs)` does what the framework does for one recorded step and returns the explanation, its bits, or the best attempt when there is none. `sample_step(state, action, rng)` draws one outcome of `step`, for trying a plan in your model. `replay(episode_id)` explains a whole recorded episode and reports the bits per choice. Read their docstrings.
"""

_EVALUATION = """
- **Final evaluation with the CWM.** The final evaluation described under Environment runs the controller with your accepted CWM, exactly as `RunController` does, except that nobody can repair the CWM there. When it cannot explain an observation, the State is rebuilt from that observation alone with `get_initial_state` and the life goes on: what the State remembered is lost. Nothing else stops the life early. Your accepted CWM is also scored by itself, on play recorded in other new worlds: the share of the steps it explains, and their cost in bits.
"""

_UNKNOWN_CELLS = """
**Unknown cells.** `render` may put the string `"__UNKNOWN__"` in place of any value inside `frame` (one cell, a row, a whole grid): that part is not compared with the real observation. Everything else must still match exactly, and `reward`, `is_done`, `available_actions` and `info` cannot be unknown. `render` decides this from the State alone. Use it for what you cannot predict yet or do not need (a counter you do not understand, the intermediate frames of an animation). The State is still carried by `step`: something you left unknown fails later as a normal mismatch if it shows in a cell you do predict, and your controller cannot rely on a part nobody checks.

When `frame` is a list of frames, `"__UNKNOWN_FRAMES__"` as its first element stands for any number of leading frames: `["__UNKNOWN_FRAMES__", last_frame]` predicts only the last one, whatever their count.

Two limits. Among the frame cells whose value changes at least once in the recorded observations, at most __UNKNOWN_SHARE__ % may be unknown; cells that never change cost nothing. The unknown parts of an observation are not counted in the observation bytes of the compactness check. An accepted validation reports `unknown_cells`.
"""
_UNKNOWN_CELLS_INTRO = """
- **Unknown cells.** Your `render` may answer `"__UNKNOWN__"` for parts of the frame you do not predict: those parts are not checked, the rest must be exact. Read "Unknown cells" in `docs/modeling_phase.md`. A frame predicted by your CWM (`EnvCWM`, a plan) contains these markers as `render` wrote them.
"""
_UNKNOWN_CELLS_FAILURE = "- `unknown_cells`: too many of the changing frame cells are unknown; `share_of_changing_cells` must be at most `allowed_share`.\n"

_IMAGE_PREVIEWS = """Up to __IMAGE_COUNT__ PNG previews are saved in `tmp/images/obs_id_<ID>.png`, selected from the first and last distinct observations encountered. `observation_images` lists the saved paths. Read them with your image tool. This folder is emptied at the start of every call; copy images elsewhere if needed. The dataset itself remains available regardless of preview cleanup.

"""
_DIAGNOSTIC_IMAGES = """ `save_image(diagnostic_id=..., which="observed", path=...)` and `which="predicted"` help compare images, when the diagnostic contains them. Size errors and code exceptions may have no image; inspect their structured data."""


_DIAGNOSTIC_IMAGE = """\n- `image`: a picture of the mismatch with four panels: Previous (the observation the step started from), Predicted, Observed, and Differences (the observed picture, dimmed where it matches the prediction and outlined in red where it does not). Open it with your image-reading tool. `save_image(path, diagnostic_id=..., which="comparison")` saves the same picture for any diagnostic."""


def _render(name: str, options: CwmConfig, *, vision: bool = False, **extra: str) -> str:
    """Render only named markers; Python braces inside Markdown remain literal."""
    planner = "PlanInCWM" in enabled_commands(options)
    values = {
        "IMAGE_PREVIEWS": _IMAGE_PREVIEWS.replace(
            "__IMAGE_COUNT__", str(options.n_tmp_images_saved_per_exploration)
        )
        if vision and options.n_tmp_images_saved_per_exploration
        else "",
        "DIAGNOSTIC_IMAGES": _DIAGNOSTIC_IMAGES if vision else "",
        "DIAGNOSTIC_IMAGE": _DIAGNOSTIC_IMAGE if vision else "",
        "PLANNING_WORKFLOW": (
            "- **Optional planning.** In the Acting Phase, define a goal in `goal.py` and "
            "use `PlanInCWM` to find an action list in the accepted CWM. You can then use "
            "that list in your controller. See `docs/plan_in_CWM.md`.\n"
            if planner
            else ""
        ),
        "ACTION_LIST_GUIDANCE": (
            "For a controller built from a planned action list, see `docs/plan_in_CWM.md`."
            if planner
            else "For a fixed sequence of actions, use `ActionListController` "
            "from `framework/action_list_controller.py`; see its docstring."
        ),
        "CALLBACK_KINDS": "CWM, controller and goal" if planner else "CWM and controller",
        "CONTROLLER_KINDS": "Controllers and goals" if planner else "Controllers",
        "GOAL_CALLBACK_LIMIT": ", and a goal's `achieved` plus `utility` together"
        if planner
        else "",
        "INITIAL_TARGET": options.n_unique_observations_in_initial_collection,
        "SIZE_RATIO": options.threshold_max_state_obs_size_ratio,
        "UNKNOWN_CELLS": _UNKNOWN_CELLS.replace("__UNKNOWN__", UNKNOWN)
        .replace("__UNKNOWN_FRAMES__", UNKNOWN_FRAMES)
        .replace("__UNKNOWN_SHARE__", f"{100 * MAX_UNKNOWN_SHARE:g}")
        if options.allow_unknown_cells
        else "",
        "UNKNOWN_CELLS_FAILURE": _UNKNOWN_CELLS_FAILURE if options.allow_unknown_cells else "",
        "STATE_ALLOWANCE": f" A State of at most {options.max_state_bytes_always_allowed} serialized bytes always passes."
        if options.max_state_bytes_always_allowed is not None
        else "",
        "EXPLORATION_ACTIONS": options.max_actions_per_RunController,
        "CALL_SECONDS": options.execution.max_seconds_per_call,
        "VALIDATION_SECONDS": options.execution.max_seconds_per_UpdateCodeWorldModel,
        "MEMORY_MB": options.execution.max_memory_mb,
        "COUNTEREXAMPLES": options.feedback.max_counterexamples,
        "DIFF_ITEMS": options.feedback.max_diff_items,
        "DIFF_REGIONS": options.feedback.max_diff_regions,
        "DIFF_CHANGES": options.feedback.max_diff_changes,
        "ERROR_CHARS": options.feedback.max_error_chars,
        "DATA_ITEMS": options.data_api.max_items,
        "DATA_BYTES": options.data_api.max_response_bytes,
        "PLANNER_ALGORITHM": options.planner.algorithm,
        "PLANNER_SECONDS": options.planner.max_seconds_per_planner_call,
        "PLANNER_CALLS": options.planner.max_cwm_calls_per_planner_call,
        "PLANNER_NODES": options.planner.max_nodes_per_planner_call,
        "PLANNER_DEPTH": options.planner.max_depth_per_planner_call,
        "LOCAL_HELPERS": (
            ""
            if not options.workspace_helpers_enabled
            else _STOCHASTIC_HELPERS
            if options.stochastic
            else _LOCAL_HELPERS
        ),
        "STATE_BYTES": options.max_state_bytes_always_allowed,
        "ACTION_CAP": action_cap_sentence(options),
        "EXPLAIN_NODES": options.max_nodes_per_explanation,
        "SURPRISE_BITS": options.max_surprise_bits_per_choice,
        **extra,
    }
    text = (_PROMPTS / name).read_text(encoding="utf-8")
    if options.stochastic and name == "modeling_phase.md":
        text += (_PROMPTS / "modeling_phase_stochastic.md").read_text(encoding="utf-8")
    if options.stochastic and name == "acting_phase.md":
        text = text.replace("# Acting Phase\n", "# Acting Phase\n" + _STOCHASTIC_ACTING, 1)
    for key, value in values.items():
        text = text.replace(f"__{key}__", str(value) if isinstance(value, str) else describe(value))
    if options.stochastic and name == "workflow.md":
        text = text.rstrip() + _STOCHASTIC_INTRO
    if options.allow_unknown_cells and name == "workflow.md":
        text = text.rstrip() + _UNKNOWN_CELLS_INTRO.replace("__UNKNOWN__", UNKNOWN)
    return text


def workspace_docs(options: CwmConfig, *, vision: bool) -> Iterator[TemplateFile]:
    for source, destination in _DOCS.items():
        if source == "plan_in_CWM.md" and "PlanInCWM" not in enabled_commands(options):
            continue
        yield TemplateFile(destination, _render(source, options, vision=vision))


def build_prompt(
    problem: BaseProblem,
    task_name: str,
    config: RunConfig,
    options: CwmConfig,
    *,
    tool_protocol: ToolProtocol,
    tool_names: list[str],
    verbalize_variant: str,
    command_script: str,
) -> str:
    from regact.protocols.cwm.templates import templates

    context = FeatureContext(problem_name=problem.name, task_name=task_name, workdir="")
    planner = "PlanInCWM" in enabled_commands(options)
    workspace_notes = "It also reads your CWM from `world_model/`."
    if planner:
        workspace_notes += (
            " The planner reads goals from `goal.py` and saves action lists under `plans/`."
        )
    return build_managed_prompt(
        problem,
        task_name,
        config,
        options,
        files=templates(
            context,
            options,
            commands={**enabled_commands(options), **reset_commands(config, problem)},
            vision=config.agent.vision,
        ),
        descriptions=_DESCRIPTIONS,
        commands={**enabled_commands(options), **reset_commands(config, problem)},
        workspace_extensions=workspace_notes,
        phase_steps=_render("workflow.md", options).rstrip()
        + (_EVALUATION if problem.evaluation_tasks(task_name) else "\n"),
        workflow_intro=_render("intro.md", options),
        initial_guidance="You begin in the **Modeling Phase**.",
        uses_model=True,
        tool_protocol=tool_protocol,
        tool_names=tool_names,
        verbalize_variant=verbalize_variant,
        command_script=command_script,
    )
