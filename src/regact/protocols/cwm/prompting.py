"""CWM-specific content; shared PromptBuilder owns agent dialect and assembly."""

from collections.abc import Iterator
from pathlib import Path

from regact.agent.capabilities import ToolProtocol
from regact.config.schema import RunConfig
from regact.features.base import FeatureContext
from regact.problems.base import BaseProblem
from regact.prompt.builder import PromptBuilder
from regact.protocols.cwm.config import CwmConfig
from regact.protocols.cwm.limits import describe
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
    "exploration.py": "the file to edit to submit your exploration controller",
    "framework/action_list_controller.py": "an action-list-to-controller class",
    "framework/control.py": "runnable for executing framework commands",
    "framework/data_api.py": "the documented data API",
    "framework/simulation.py": "example of local simulation/debugging helpers",
    "goal.py": "the file to write goals before using the PlanInCWM command",
    "world_model/model_parser.py": 'define your "parser: observation (dictionary) -> State"',
    "world_model/model_render.py": 'define your "render: state -> observation (dictionary)"',
    "world_model/model_state.py": 'define your class "State"',
    "world_model/model_transition.py": 'define your "step: state, action -> state"',
}
_LOCAL_HELPERS = """
## Local simulation helpers

`framework/simulation.py` contains `EnvCWM`, `make_cwm_env` and `run_controller`. Read their docstrings for examples. They use the current workspace CWM; they do not validate or submit it, update the dataset, or perform real actions.

`EnvCWM(initial_state=state)` requires an explicit State and never queries the dataset. `make_cwm_env()` instead loads the recorded starting observation once and parses it; passing initial_state skips that lookup. Both return States from reset()/step(action), or full observation dictionaries with obs_mode=True. Reset restores the starting State without taking a real action.

`python framework/simulation.py --max-actions 20` runs exploration.py locally and prints each action and State. Local execution uses the shell tool's timeout, not isolated-callback limits. Start a new script after edits. Submitted callbacks must supply a State explicitly, not load data through the factory.
"""
_TERMINAL_EXAMPLES = {
    "EDIT_EXAMPLE": "python inspect_data.py",
    "SCRIPT_PATH": "inspect_data.py",
    "SCRIPT_IMPORT": "from framework import data_api\nprint(data_api.summary())",
    "CONTROLLER_PATH": "exploration.py",
    "ACTION_EXAMPLE": "chosen_action",
    "SCRIPT_DIR": ".",
}


def _render(name: str, options: CwmConfig, **extra: str) -> str:
    """Render only named markers; Python braces inside Markdown remain literal."""
    values = {
        "INITIAL_TARGET": options.n_unique_observations_in_initial_collection,
        "SIZE_RATIO": options.threshold_max_state_obs_size_ratio,
        "EXPLORATION_ACTIONS": options.max_actions_per_exploration,
        "IMAGE_COUNT": options.n_tmp_images_saved_per_exploration,
        "CALL_SECONDS": options.execution.max_seconds_per_call,
        "VALIDATION_SECONDS": options.execution.max_seconds_per_UpdateCodeWorldModel,
        "EPISODE_SECONDS": options.execution.max_seconds_per_episode,
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


def workspace_docs(options: CwmConfig) -> Iterator[TemplateFile]:
    for source, destination in _DOCS.items():
        yield TemplateFile(destination, _render(source, options))


def _workspace_tree(files: list[TemplateFile], helper_paths: set[str]) -> str:
    """List exactly the generated files, including optional/problem-provided ones."""
    descriptions = {"framework/__init__.py": "empty"}
    for file in files:
        descriptions[file.relpath] = _DESCRIPTIONS.get(
            file.relpath,
            "game action/observation helpers"
            if file.relpath in helper_paths
            else "empty"
            if file.relpath.endswith("/__init__.py")
            else "provided workspace file",
        )
    rows = []
    parent = None
    for path, description in sorted(descriptions.items()):
        directory, _, filename = path.rpartition("/")
        if directory and directory != parent:
            rows.append(directory + "/")
        parent = directory
        label = "  " + filename if directory else filename
        rows.append(f"{label:<34} # {description}")
    return "\n".join(rows)


def build_prompt(
    problem: BaseProblem,
    task_name: str,
    config: RunConfig,
    options: CwmConfig,
    *,
    tool_protocol: ToolProtocol,
    tool_names: list[str],
    verbalize_variant: str,
) -> str:
    from regact.protocols.cwm.templates import templates

    context = FeatureContext(problem_name=problem.name, task_name=task_name, workdir="")
    helpers = problem.helper_templates(
        task_name,
        info_mode=config.problem.info_mode,
        helper=config.problem.helper,
        direct_interaction=False,
    )
    tree = _workspace_tree([*templates(context, options), *helpers], {f.relpath for f in helpers})
    return PromptBuilder().assemble_system_prompt(
        [
            _render("system.md", options, WORKSPACE_TREE=tree),
            problem.build_prompt(
                task_name,
                direct_interaction=False,
                info_mode=config.problem.info_mode,
                obs_mode=config.problem.obs_mode,
            ),
            _render("workflow.md", options),
        ],
        tool_protocol=tool_protocol,
        tool_names=tool_names,
        verbalize_variant=verbalize_variant,
        terminal_examples=_TERMINAL_EXAMPLES,
    )
