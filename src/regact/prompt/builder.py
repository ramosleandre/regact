"""Prompt assembly.

Assembly logic only — no prompt text lives here. The static framework brief is in
``prompt/system.md``; the game section comes from ``problem.build_prompt`` (per task
and info level); each feature contributes a fragment; the control and lifecycle
blocks are selected by capability/enum. Everything static for a run goes in the
SYSTEM prompt (stable across the run's turns, so it caches); the FIRST MESSAGE
carries only the dynamic first observation. To change wording, edit the markdown
(or the problem/feature), not this file. Empty sections are dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from regact.agent.capabilities import ToolProtocol, uses_control_cli
from regact.config.schema import InfoMode, Lifecycle, ObsMode
from regact.features.base import Feature, FeatureContext

if TYPE_CHECKING:
    from regact.features.controller import Controller
    from regact.problems.base import BaseProblem

_PROMPT_DIR = Path(__file__).parent
_SYSTEM_MD = _PROMPT_DIR / "system.md"
_LIFECYCLE_MD = {
    Lifecycle.SINGLE_INSTANCE: _PROMPT_DIR / "lifecycle_single.md",
    Lifecycle.MULTI_INSTANCE: _PROMPT_DIR / "lifecycle_multi.md",
}
# One fragment per bash-only dialect (native/client_cli read none - see _control_channel_block).
# Each teaches the same one-command-per-turn loop; they differ only in the tool-call markup.
_TERMINAL_MD = {
    "bash_block": _PROMPT_DIR / "bash_block_terminal.md",  # swegrid: fenced ```bash block
    "hermes_xml": _PROMPT_DIR / "hermes_xml_terminal.md",  # Qwen/hermes <tool_call> markup
    "glm": _PROMPT_DIR / "glm_terminal.md",  # GLM <tool_call>Bash<arg_key>/<arg_value> markup
}
_FEATURES_INTRO = "# Features :\n\nYou are given the following features to help you."

# Tier-2 of the empty_response fix (opt-in, for the A/B vs the in-loop nudge alone):
# silent thinkers lose state because their hidden reasoning is stripped between turns,
# degenerate to empty answers, and get killed. Teaching up-front verbalization makes
# them persist state in the visible answer. Selectable variants (``verbalize_variant``)
# so the strategy-sweep can A/B their strength: ``v1`` = forceful prose demand,
# ``v2`` = a mandated THOUGHTS: structure. Both open with the same header so a test /
# the sweep can detect that the hint is present.
_VERBALIZE_HEADER = "# Keep your working state visible"
_VERBALIZE_VARIANTS = {
    "v1": (
        f"{_VERBALIZE_HEADER}\n\n"
        "Your private reasoning is NOT carried over between turns - only your visible answer "
        "text and your tool calls persist. So you MUST externalize it: before every tool call, "
        "write two or three plain sentences stating what you just observed, your current plan, "
        "and why this next action. Never act on silent reasoning alone - the harness discards it "
        "and you will lose track of your own progress across turns."
    ),
    "v2": (
        f"{_VERBALIZE_HEADER}\n\n"
        "Your private reasoning is NOT carried over between turns - only your visible answer "
        "persists. So every turn's answer MUST follow this exact structure:\n"
        "1. a line `THOUGHTS: <one sentence summarizing your current state and plan>`\n"
        "2. then your single tool call.\n"
        "An answer with no THOUGHTS line, or with only a tool call, is invalid - always write "
        "THOUGHTS first so your state survives into the next turn."
    ),
}


class PromptBuilder:
    """Compose the system prompt (everything static) and the first user message."""

    def build_system_prompt(
        self,
        problem: BaseProblem,
        task_name: str,
        features: list[Feature],
        *,
        controller: Controller | None = None,
        lifecycle: Lifecycle,
        info_mode: InfoMode = InfoMode.INFORMATIVE,
        obs_mode: ObsMode = ObsMode.RAW,
        tool_protocol: ToolProtocol = "native",
        tool_names: list[str] | None = None,
        verbalize_variant: str = "off",
    ) -> str:
        """The full static brief: framework role + game + controller + features + control +
        lifecycle.

        The always-on controller's fragment is a core section (its own working approach);
        the OPTIONAL features get the generic "# Features" intro. Stable across a run's turns
        (cache-friendly); the dynamic observation is sent separately as the first message.
        """
        ctx = FeatureContext(problem_name=problem.name, task_name=task_name, workdir="")
        sections = [
            _SYSTEM_MD.read_text(encoding="utf-8"),
            problem.build_prompt(task_name, info_mode=info_mode, obs_mode=obs_mode),
        ]
        if controller is not None and (core := controller.prompt_fragment(ctx)):
            sections.append(core)  # the controller is core, not under "# Features"
        fragments = [frag for f in features if (frag := f.prompt_fragment(ctx))]
        if fragments:  # generic intro, then each feature describes its own deliverable
            sections.append(_FEATURES_INTRO)
            sections += fragments
        sections.append(_control_channel_block(tool_protocol, tool_names or []))
        sections.append(_LIFECYCLE_MD[lifecycle].read_text(encoding="utf-8"))
        if (hint := _VERBALIZE_VARIANTS.get(verbalize_variant)) is not None:
            sections.append(hint)
        return "\n\n".join(s.strip() for s in sections if s and s.strip())

    def build_first_message(self, rendered_obs: str | None = None) -> str:
        """The first user message: the first observation (for reference) + a generic, agnostic
        start. The how (controller, tools, approach) all lives in the system prompt."""
        start = "Begin working on the task described in your instructions above."
        if rendered_obs:
            header = f"This is the first observation of the game, for reference. {start}"
            return f"{header}\n\n{rendered_obs}"
        return start


def _control_channel_block(
    tool_protocol: ToolProtocol,
    tool_names: list[str],
) -> str:
    """How the agent invokes tools - selected by the agent's ``tool_protocol``, never by a
    feature or a concrete agent name.

    Generic: lists the tool NAMES (from the run's tools) and the invocation the protocol
    supports; it never imports a tool or a feature type. A bash-only dialect (bash_block,
    hermes_xml) reads its whole terminal fragment (one-command-per-turn loop + shell idioms)
    and folds submit/exit in; the framework actions are the same shell commands in either.
    """
    if not tool_names:
        return ""
    # native: in-process loop tools, so the model calls them directly. Every other protocol reaches
    # them over the workdir control CLI - the SAME split task.py binds on via uses_control_cli (a
    # channel taught here but not bound there is the seam bug).
    if not uses_control_cli(tool_protocol):
        return f"# Framework tools\n\nCall the framework tools directly: {', '.join(tool_names)}."
    if tool_protocol in _TERMINAL_MD:  # bash-only dialect: fold submit/exit into its fragment
        commands = "\n".join(f"python framework/control.py {name}" for name in tool_names)
        terminal = _TERMINAL_MD[tool_protocol].read_text(encoding="utf-8")
        return terminal.replace("{control_commands}", commands).strip()
    # client_cli (Claude/codex): a plain list of the control commands
    lines = "\n".join(f"- `python framework/control.py {name}`" for name in tool_names)
    return (
        "# Framework tools\n\n"
        "Run a framework tool from your working directory; each prints its result "
        "(e.g. your score) to stdout:\n\n"
        f"{lines}"
    )
