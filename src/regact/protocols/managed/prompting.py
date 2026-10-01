"""Common agent brief; protocols extend only the workspace and workflow.

Role, terminal dialect, game instructions, dataset access, image previews and
execution rules have one source. Protocols supply their generated files and
workflow steps, rather than copying and progressively diverging system prompts.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from regact.agent.capabilities import ToolProtocol
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
    files: Iterable[TemplateFile],
    helper_paths: set[str],
    descriptions: Mapping[str, str] | None = None,
) -> str:
    """List actual generated files, with protocol-supplied descriptions as additions."""
    known = {**_DESCRIPTIONS, **(descriptions or {})}
    entries = {"framework/__init__.py": "empty"}
    for file in files:
        entries[file.relpath] = known.get(
            file.relpath,
            "game action/observation helpers"
            if file.relpath in helper_paths
            else "empty"
            if file.relpath.endswith("/__init__.py")
            else "provided workspace file",
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


def image_preview_instructions(count: int) -> str:
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
    workspace_extensions: str = "",
    workflow_steps: str | None = None,
    workflow_intro: str = "",
    initial_guidance: str = "",
) -> str:
    """Assemble the managed baseline, with optional protocol-specific additions.

    Files must be the same templates used to initialize this task's workdir.
    Problem helpers and the game section use identical calls for every protocol.
    """
    helpers = problem.helper_templates(
        task_name,
        info_mode=config.problem.info_mode,
        helper=config.problem.helper,
        direct_interaction=False,
    )
    tree = workspace_tree([*files, *helpers], {f.relpath for f in helpers}, descriptions)
    workspace = _render(
        "workspace.md",
        WORKSPACE_TREE=tree,
        WORKSPACE_EXTENSIONS=workspace_extensions.strip() + " " if workspace_extensions else "",
        IMAGE_PREVIEWS=image_preview_instructions(options.n_tmp_images_saved_per_exploration),
    )
    workflow = _render(
        "workflow.md",
        PROTOCOL_INTRO=workflow_intro.strip() + "\n\n" if workflow_intro else "",
        INITIAL_TARGET=options.n_unique_observations_in_initial_collection,
        INITIAL_GUIDANCE=initial_guidance,
        PROTOCOL_STEPS=(workflow_steps if workflow_steps is not None else _render("vanilla_steps.md")).strip(),
        LIFECYCLE=lifecycle_instructions(config, problem),
    )
    return PromptBuilder().assemble_system_prompt(
        [
            _render("role.md"),
            workspace,
            problem.build_prompt(
                task_name,
                direct_interaction=False,
                info_mode=config.problem.info_mode,
                obs_mode=config.problem.obs_mode,
            ),
            workflow,
        ],
        tool_protocol=tool_protocol,
        tool_names=tool_names,
        verbalize_variant=verbalize_variant,
        terminal_examples=_TERMINAL_EXAMPLES,
        command_script=command_script,
    )


def lifecycle_instructions(config, problem):
    if config.problem.lifecycle is Lifecycle.SINGLE_INSTANCE:
        start = "RunController continues from the current live observation, including where initial random collection stopped."
    else:
        start = "Each RunController starts a fresh environment from the initial state."
    resets = " ".join(
        f"You can {description[0].lower()}{description[1:].rstrip('.')} with the {name} command."
        for name, description in problem.reset_commands().items()
    )
    return (
        start + " Every call creates a fresh controller; its private memory is not carried over. "
        "An environment reporting is_done=True stops this controller call. "
        + resets
    )
