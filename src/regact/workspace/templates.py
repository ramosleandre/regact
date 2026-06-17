"""A file to scaffold into the agent's workdir.

A neutral workspace primitive: both features (``Feature.templates``) and problems
(``BaseProblem.helper_templates``) emit these, and ``Workspace.bootstrap`` writes
them. Living here (not in ``features/``) keeps problems decoupled from features.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemplateFile:
    """A file dropped into the agent's workdir at a relative path."""

    relpath: str  # e.g. "code_library/world_model.py"
    content: str  # rendered template body (signatures the agent fills)
