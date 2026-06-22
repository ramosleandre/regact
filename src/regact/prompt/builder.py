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
from typing import TYPE_CHECKING, Literal

from regact.config.schema import InfoMode, Lifecycle
from regact.features.base import Feature, FeatureContext

if TYPE_CHECKING:
    from regact.problems.base import BaseProblem

_PROMPT_DIR = Path(__file__).parent
_SYSTEM_MD = _PROMPT_DIR / "system.md"
_LIFECYCLE_MD = {
    Lifecycle.SINGLE_INSTANCE: _PROMPT_DIR / "lifecycle_single.md",
    Lifecycle.MULTI_INSTANCE: _PROMPT_DIR / "lifecycle_multi.md",
}
_CLOSING = "Do not stop until you have feedback confirming you have solved the game."


class PromptBuilder:
    """Compose the system prompt (everything static) and the first user message."""

    def build_system_prompt(
        self,
        problem: BaseProblem,
        task_name: str,
        features: list[Feature],
        *,
        lifecycle: Lifecycle,
        info_mode: InfoMode = InfoMode.INFORMATIVE,
        control_actions: Literal["native_tools", "client_cli"] = "native_tools",
        tool_names: list[str] | None = None,
    ) -> str:
        """The full static brief: framework role + game + features + control + lifecycle.

        Stable across a run's turns (cache-friendly); the dynamic observation is sent
        separately as the first message.
        """
        ctx = FeatureContext(problem_name=problem.name, task_name=task_name, workdir="")
        sections = [
            _SYSTEM_MD.read_text(encoding="utf-8"),
            problem.build_prompt(task_name, info_mode=info_mode),
        ]
        sections += [f.prompt_fragment(ctx) or "" for f in features]
        sections.append(_control_channel_block(control_actions, tool_names or []))
        sections.append(_LIFECYCLE_MD[lifecycle].read_text(encoding="utf-8"))
        sections.append(_CLOSING)
        return "\n\n".join(s.strip() for s in sections if s and s.strip())

    def build_first_message(self, rendered_obs: str | None = None) -> str:
        """The first user message: an optional pre-rendered observation, else a generic start."""
        if rendered_obs:
            return (
                "This is the first observation of the game. Explore it, build a policy, "
                "and keep going until you reach the end.\n\n"
                f"{rendered_obs}"
            )
        return (
            "Begin: explore the environment, then write, test, and submit your controller. "
            "Keep going until you win."
        )


def _control_channel_block(
    control_actions: Literal["native_tools", "client_cli"], tool_names: list[str]
) -> str:
    """How to invoke framework tools — depends on the agent, not the feature.

    Generic: lists the tool NAMES (from the run's tools) and the invocation the backend
    supports; it never imports a tool or a feature type.
    """
    if not tool_names:
        return ""
    if control_actions == "client_cli":
        lines = "\n".join(f"- `python framework/control.py {name}`" for name in tool_names)
        return (
            "## Framework tools\n\n"
            "Run a framework tool from your working directory; each prints its result "
            "(e.g. your score) to stdout:\n\n"
            f"{lines}"
        )
    return f"## Framework tools\n\nCall the framework tools directly: {', '.join(tool_names)}."
