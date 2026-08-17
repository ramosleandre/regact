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

from regact.agent.capabilities import ToolProtocol
from regact.config.schema import InfoMode, Lifecycle
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
}
_FEATURES_INTRO = "# Features :\n\nYou are given the following features to help you."


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
        tool_protocol: ToolProtocol = "native",
        tool_names: list[str] | None = None,
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
            problem.build_prompt(task_name, info_mode=info_mode),
        ]
        if controller is not None and (core := controller.prompt_fragment(ctx)):
            sections.append(core)  # the controller is core, not under "# Features"
        fragments = [frag for f in features if (frag := f.prompt_fragment(ctx))]
        if fragments:  # generic intro, then each feature describes its own deliverable
            sections.append(_FEATURES_INTRO)
            sections += fragments
        sections.append(_control_channel_block(tool_protocol, tool_names or []))
        sections.append(_LIFECYCLE_MD[lifecycle].read_text(encoding="utf-8"))
        return "\n\n".join(s.strip() for s in sections if s and s.strip())

    def build_first_message(self, rendered_obs: str | None = None) -> str:
        """The first user message: the first observation (for reference) + a generic, agnostic
        start. The how (controller, tools, approach) all lives in the system prompt."""
        start = (
            "Begin working on the task described in your instructions above. "
            "Keep going until you solve the game."
        )
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
    if tool_protocol in _TERMINAL_MD:
        commands = "\n".join(f"python framework/control.py {name}" for name in tool_names)
        terminal = _TERMINAL_MD[tool_protocol].read_text(encoding="utf-8")
        return terminal.replace("{control_commands}", commands).strip()
    if tool_protocol == "client_cli":
        lines = "\n".join(f"- `python framework/control.py {name}`" for name in tool_names)
        return (
            "## Framework tools\n\n"
            "Run a framework tool from your working directory; each prints its result "
            "(e.g. your score) to stdout:\n\n"
            f"{lines}"
        )
    return f"## Framework tools\n\nCall the framework tools directly: {', '.join(tool_names)}."
