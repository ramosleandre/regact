"""Common agent brief; protocols extend only the workspace and workflow.

Role, terminal dialect, environment, dataset access, image previews and execution rules have
one source. The environment section is the same for every game: a problem contributes only
its action format line. Protocols and switches add their own files and workflow steps.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from regact.agent.capabilities import ToolProtocol, uses_control_cli
from regact.config.schema import Lifecycle, RunConfig
from regact.problems.base import BaseProblem
from regact.prompt.builder import PromptBuilder
from regact.workspace.templates import TemplateFile

_PROMPTS = Path(__file__).with_name("prompts")
_DESCRIPTIONS = {
    "controller.py": "the file to edit before calling RunController",
    "framework/action_list_controller.py": "an action-list-to-controller class",
    "framework/commands.py": "runnable for executing framework commands",
    "framework/data_api.py": "the documented data API",
}
_TERMINAL_EXAMPLES = {
    "EDIT_EXAMPLE": "python inspect_data.py",
    "SCRIPT_PATH": "inspect_data.py",
    "SCRIPT_IMPORT": "from framework import data_api\nprint(data_api.summary())",
    "CONTROLLER_PATH": "controller.py",
    "ACTION_EXAMPLE": "chosen_action",
    "SCRIPT_DIR": ".",
}


def _render(name: str, **values: Any) -> str:
    """Substitute named markers without interpreting Python braces in examples."""
    text = (_PROMPTS / name).read_text(encoding="utf-8")
    for key, value in values.items():
        rendered = "unlimited" if value is None else str(value)
        text = text.replace(f"__{key}__", rendered)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def workspace_tree(
    files: Iterable[TemplateFile], descriptions: Mapping[str, str] | None = None
) -> str:
    """List actual generated files, with protocol-supplied descriptions as additions."""
    known = {**_DESCRIPTIONS, **(descriptions or {})}
    entries = {"framework/__init__.py": "empty"}
    for file in files:
        entries[file.relpath] = known.get(
            file.relpath,
            "empty" if file.relpath.endswith("/__init__.py") else "provided workspace file",
        )
    rows = []
    parent = None
    for path, description in sorted(entries.items()):
        directory, _, filename = path.rpartition("/")
        if directory and directory != parent:
            rows.append(directory + "/")
        parent = directory
        label = "  " + filename if directory else filename
        rows.append(f"{label:<34} # {description}")
    return "\n".join(rows)


def action_cap_sentence(options: Any) -> str:
    cap = options.max_actions_per_RunController
    return f" A call plays at most {cap} actions." if cap is not None else ""


def image_preview_instructions(options: Any, *, can_view_images: bool = True) -> str:
    if not options.exp_buffer:
        return (
            "After every command, the current observation is also saved as "
            "`tmp/images/final.png`; open it with your image-reading tool. The file is "
            "overwritten at every command."
            if can_view_images
            else ""
        )
    count = options.n_tmp_images_saved_per_exploration
    if not can_view_images:
        return (
            "Your tools cannot display images: read observations as arrays through "
            "`framework.data_api` and analyse them in code."
        )
    if count == 0:
        return (
            "Automatic image previews are disabled. Use `framework.data_api.save_image` "
            "to save recorded images, then open them with your image-reading tool."
        )
    return (
        f"After a `RunController` call, up to {count} observation images are saved as "
        "`tmp/images/obs_id_<ID>.png`, selected from the first and last distinct observations "
        "encountered. The feedback lists their paths; open them with your image-reading tool. "
        "This temporary folder is cleared at the start of every call, including a refused "
        "one; copy images elsewhere if you need to keep them. Recorded data remains "
        "available through `framework.data_api`."
    )


def printing_instructions(problem: BaseProblem) -> str:
    """How a result prints the observation without the experience buffer (observation_text)."""
    if problem.frame_cells == "hex":
        return (
            "The observation's grid is printed first, one row per line and one hex digit per "
            "cell (`a` = 10 ... `f` = 15), then the observation as JSON, where a placeholder "
            "marks the grid's place in `frame`; your controller receives it as integers."
        )
    return "The observation is printed as JSON, one line per grid row."


def build_prompt(
    problem: BaseProblem,
    task_name: str,
    config: RunConfig,
    options: Any,
    *,
    files: Iterable[TemplateFile],
    tool_protocol: ToolProtocol,
    tool_names: list[str],
    verbalize_variant: str,
    command_script: str,
    descriptions: Mapping[str, str] | None = None,
    commands: Mapping[str, str] | None = None,
    workspace_extensions: str = "",
    workflow_intro: str = "",
    phase_steps: str = "",
    initial_guidance: str = "",
    uses_model: bool = False,
) -> str:
    """Assemble the managed brief: one Role and Environment section for every game and
    protocol, with each switch adding only its own lines.

    Files must be the same templates used to initialize this task's workdir.
    """
    buffer = options.exp_buffer
    workspace = _render(
        "workspace.md",
        WORKSPACE_TREE=workspace_tree(files, descriptions),
        WORKSPACE_EXTENSIONS=workspace_extensions.strip() + " " if workspace_extensions else "",
        IMAGE_PREVIEWS=image_preview_instructions(options, can_view_images=config.agent.vision),
    )
    steps = []
    if buffer:
        steps.append(
            _render(
                "recorded_experience.md",
                INITIAL_DATASET=(
                    "At the start the dataset holds only the first observation."
                    if options.n_unique_observations_in_initial_collection <= 1
                    else "A random policy already filled the dataset, targeting "
                    f"{options.n_unique_observations_in_initial_collection} unique observations "
                    "(possibly fewer if a collection limit was reached)."
                ),
                INITIAL_GUIDANCE=" " + initial_guidance if initial_guidance else "",
            )
        )
    if phase_steps:
        steps.append(phase_steps.strip())
    steps.append(
        _render(
            "write_controller.md",
            INPUT="state" if uses_model else "obs",
            INPUT_DESCRIPTION=(
                "Both receive the State of your accepted CWM: built by `get_initial_state`, then "
                "advanced by `step` after each action."
                if uses_model
                else "Both receive the current observation dictionary."
            ),
        )
    )
    if not uses_model:
        steps.append(
            _render(
                "run_controller.md",
                RUN_CONTROLLER="`python framework/commands.py RunController`"
                if uses_control_cli(tool_protocol)
                else "the `RunController` tool",
                ACTION_CAP=action_cap_sentence(options),
            )
        )
    steps.append(
        _render(
            "returns_buffer.md",
            COUNTEREXAMPLE="the counterexample when the CWM was contradicted, " if uses_model else "",
        )
        if buffer
        else _render("returns_no_buffer.md", PRINTING=printing_instructions(problem))
    )
    workflow = _render(
        "workflow.md",
        PROTOCOL_INTRO=workflow_intro.strip() + "\n\n" if workflow_intro else "",
        PROTOCOL_STEPS="\n".join(steps),
        LIFECYCLE=lifecycle_instructions(config),
        OBJECTIVE_NOTE=" " + config.objective_note if config.objective_note else "",
        CALLBACKS="Controllers and the CWM" if uses_model else "Controllers",
        INACCESSIBLE="the dataset, " if buffer else "",
    )
    return PromptBuilder().assemble_system_prompt(
        [
            _render("role.md"),
            workspace,
            _render("environment.md", ACTION_FORMAT=problem.action_format()),
            workflow,
        ],
        tool_protocol=tool_protocol,
        tool_names=tool_names,
        verbalize_variant=verbalize_variant,
        terminal_examples=_TERMINAL_EXAMPLES,
        command_script=command_script,
        command_descriptions=commands,
    )


def reset_commands(config, problem):
    """Reset commands exist only in single_instance: a multi_instance RunController starts from a
    fresh environment anyway, so a reset there would only spend an action."""
    if config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE:
        return problem.reset_commands()
    return {}


def lifecycle_instructions(config):
    if config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE:
        start = (
            "RunController continues from the current observation: there is one environment for "
            "the whole task, and the action count carries across calls."
        )
    else:
        start = "Each RunController starts a fresh environment from the initial observation."
    return start + " Every call creates a fresh controller; its private memory is not carried over."
