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
from regact.protocols.managed.prompting import build_prompt as build_managed_prompt
from regact.protocols.managed.prompting import reset_commands
from regact.workspace.templates import TemplateFile

_PROMPTS = Path(__file__).with_name("prompts")
_DOCS = {
    "CWM_modeling_phase.md": "docs/CWM_modeling_phase.md",
    "active_exploration_phase.md": "docs/active_exploration_phase.md",
    "plan_in_CWM.md": "docs/plan_in_CWM.md",
}
_DESCRIPTIONS = {
    "docs/CWM_modeling_phase.md": "how to implement, validate and repair the CWM",
    "docs/active_exploration_phase.md": "how to actively interact with the environment",
    "docs/plan_in_CWM.md": "(optional) how to use the CWM-dependent planner tool",
    "framework/cwm_env.py": "EnvCWM and its recorded-start initialization helper",
    "simulate.py": "editable script for optional local CWM simulation",
    "goal.py": "the file to write goals before using the PlanInCWM command",
    "world_model/model_parser.py": 'define your "parser: observation (dictionary) -> State"',
    "world_model/model_render.py": 'define your "render: state -> observation (dictionary)"',
    "world_model/model_state.py": 'define your class "State"',
    "world_model/model_transition.py": 'define your "step: state, action -> state"',
}
_LOCAL_HELPERS = """
## Local simulation helpers

`framework/cwm_env.py` provides `EnvCWM` and `make_cwm_env`. Read their docstrings for arguments and examples. They use the current workspace CWM; they do not validate or submit it, update the dataset, or perform real actions.

`EnvCWM(initial_state=state)` requires an explicit State and never queries the dataset. `make_cwm_env()` instead loads the next controller's starting observation once and parses it: the current observation in single-instance mode, or the original initial observation in multi-instance mode. Passing initial_state skips that lookup. Both return States from reset()/step(action), or full observation dictionaries with obs_mode=True. Reset restores the starting State without taking a real action.

`python simulate.py --max-actions 20` runs controller.py locally and prints each action and State. Edit simulate.py freely to inspect predictions or try different starting states. Local execution uses the shell tool's timeout, not isolated-callback limits. Start a new script after edits. Validate changed CWM files with UpdateCodeWorldModel before submitting a real exploration. Submitted callbacks must supply a State explicitly, not load data through the factory.
"""

_IMAGE_PREVIEWS = """Up to __IMAGE_COUNT__ PNG previews are saved in `tmp/images/obs_id_<ID>.png`, selected from the first and last distinct observations encountered. `observation_images` lists the saved paths. Read them with your image tool. This folder is emptied at every new submission, including a refused one; copy images elsewhere if needed. The dataset itself remains available regardless of preview cleanup.

"""
_DIAGNOSTIC_IMAGES = """ `save_image(diagnostic_id=..., which="observed", path=...)` and `which="predicted"` help compare images, when the diagnostic contains them. Size errors and code exceptions may have no image; inspect their structured data."""


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
        "PLANNING_WORKFLOW": (
            "- **Optional planning.** In Active Exploration, define a goal in `goal.py` and "
            "use `PlanInCWM` to find an action list in the accepted CWM. You can then use "
            "that list in your exploration controller. See `docs/plan_in_CWM.md`.\n"
            if planner
            else ""
        ),
        "ACTION_LIST_GUIDANCE": (
            "For a controller built from a planned action list, see `docs/plan_in_CWM.md`."
            if planner
            else "For a fixed sequence of actions, use `ExplorationControllerFromListActions` "
            "from `framework/action_list_controller.py`; see its docstring."
        ),
        "CALLBACK_KINDS": "CWM, controller and goal" if planner else "CWM and controller",
        "CONTROLLER_KINDS": "Controllers and goals" if planner else "Controllers",
        "GOAL_CALLBACK_LIMIT": ", and a goal's `achieved` plus `utility` together"
        if planner
        else "",
        "INITIAL_TARGET": options.n_unique_observations_in_initial_collection,
        "SIZE_RATIO": options.threshold_max_state_obs_size_ratio,
        "EXPLORATION_ACTIONS": options.max_actions_per_exploration,
        "CALL_SECONDS": options.execution.max_seconds_per_call,
        "VALIDATION_SECONDS": options.execution.max_seconds_per_UpdateCodeWorldModel,
        "MEMORY_MB": options.execution.max_memory_mb,
        "COUNTEREXAMPLES": options.feedback.max_counterexamples,
        "DIFF_ITEMS": options.feedback.max_diff_items,
        "ERROR_CHARS": options.feedback.max_error_chars,
        "DATA_ITEMS": options.data_api.max_items,
        "DATA_BYTES": options.data_api.max_response_bytes,
        "PLANNER_ALGORITHM": options.planner.algorithm,
        "PLANNER_SECONDS": options.planner.max_seconds_per_planner_call,
        "PLANNER_CALLS": options.planner.max_cwm_calls_per_planner_call,
        "PLANNER_NODES": options.planner.max_nodes_per_planner_call,
        "PLANNER_DEPTH": options.planner.max_depth_per_planner_call,
        "LOCAL_HELPERS": _LOCAL_HELPERS if options.workspace_helpers_enabled else "",
        **extra,
    }
    text = (_PROMPTS / name).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(f"__{key}__", str(value) if isinstance(value, str) else describe(value))
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
    if options.workspace_helpers_enabled:
        workspace_notes += (
            " `simulate.py` is an optional starter script for local simulation; "
            "adapt it or write your own scripts."
        )
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
        workspace_extensions=workspace_notes,
        workflow_steps=_render("workflow.md", options),
        workflow_intro=_render("intro.md", options),
        initial_guidance="You begin in **CWM Modeling**.",
        tool_protocol=tool_protocol,
        tool_names=tool_names,
        verbalize_variant=verbalize_variant,
        command_script=command_script,
    )
