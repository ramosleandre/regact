"""Prompt assembly.

Assembly logic only — no prompt text lives here. The static system prompt is in
``prompt/system.md``; the game section comes from ``problem.build_prompt`` (built
per task and info level); each feature contributes a fragment. To change wording,
edit the markdown (or the problem/feature), not this file. Empty sections dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from regact.config.schema import InfoMode
from regact.features.base import Feature, FeatureContext

if TYPE_CHECKING:
    from regact.problems.base import BaseProblem

_SYSTEM_MD = Path(__file__).with_name("system.md")


class PromptBuilder:
    """Compose the system prompt and the first user message from parts."""

    def build_system_prompt(self) -> str:
        """Return ``prompt/system.md`` (static, task- and problem-agnostic, cache-friendly)."""
        return _SYSTEM_MD.read_text(encoding="utf-8")

    def build_first_message(
        self,
        problem: BaseProblem,
        task_name: str,
        features: list[Feature],
        *,
        info_mode: InfoMode = InfoMode.INFORMATIVE,
    ) -> str:
        """Layer: the game prompt + per-feature fragments. Empty sections are dropped."""
        ctx = FeatureContext(
            problem_name=problem.name,
            task_name=task_name,
            workdir="",
        )
        sections: list[str] = [problem.build_prompt(task_name, info_mode=info_mode)]
        for feature in features:
            fragment = feature.prompt_fragment(ctx)
            if fragment:
                sections.append(fragment)
        return "\n\n".join(section.strip() for section in sections if section and section.strip())
