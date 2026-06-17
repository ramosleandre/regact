"""Workdir bootstrap.

Lays out the agent's working directory. The common base is **agnostic**: the
directory tree (``code_library/``, ``knowledge_base/``, ``framework/``) and the
env/lifecycle-specific ``framework/make_env.py``. Everything controller-specific
(the ``solution.py`` stub, the example controller, the contract prompt) belongs
to ``ControllerFeature`` and arrives as feature templates layered on top.
"""

from __future__ import annotations

import os

from regact.config.schema import Lifecycle
from regact.features.base import Feature, FeatureContext
from regact.workspace.templates import TemplateFile

# ``make_env`` connects to the env server the loop launched. The server URL and
# game id are baked in at bootstrap so the workdir is self-contained. Under
# MULTI_INSTANCE each call yields a fresh episode; under SINGLE_INSTANCE the one
# shared handle is returned and the agent uses ``env.reset()`` for a level reset.
# This is base (not a feature) because it depends on the env/lifecycle, not the
# capability the agent is building.
_MAKE_ENV_MULTI = '''\
"""Connect to the environment server (a fresh episode per make_env()).

You never import the game; this returns an EnvClient with a gym-like interface.
"""

from regact.envclient.client import EnvClient

_BASE_URL = {base_url!r}
_GAME_ID = {game_id!r}


def make_env(record_frames: bool = False):
    env = EnvClient.connect(_BASE_URL, _GAME_ID)
    env.reset()
    return env
'''

_MAKE_ENV_SINGLE = '''\
"""Connect to the environment server (one shared handle for the whole run).

The game can be made only once. Call ``env.reset()`` for a level reset; the
action count carries across resets. You never import the game.
"""

from regact.envclient.client import EnvClient

_BASE_URL = {base_url!r}
_GAME_ID = {game_id!r}
_HANDLE = None


def make_env(record_frames: bool = False):
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = EnvClient.connect(_BASE_URL, _GAME_ID)
        _HANDLE.reset()
    return _HANDLE
'''


class Workspace:
    """The agent's working directory."""

    def __init__(self, root: str) -> None:
        self.root = root

    def bootstrap(
        self,
        features: list[Feature],
        *,
        problem_name: str,
        task_name: str,
        env_base_url: str,
        game_id: str,
        lifecycle: Lifecycle,
        helper_templates: list[TemplateFile] | None = None,
    ) -> None:
        """Create the agnostic base, then drop problem helpers + every feature's templates."""
        os.makedirs(self.root, exist_ok=True)
        for sub in ("code_library", "knowledge_base", "framework"):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

        self._write("framework/__init__.py", "")
        template = _MAKE_ENV_SINGLE if lifecycle is Lifecycle.SINGLE_INSTANCE else _MAKE_ENV_MULTI
        self._write(
            "framework/make_env.py",
            template.format(base_url=env_base_url, game_id=game_id),
        )

        # Problem-specific helpers (e.g. ARC action constants) — import-free.
        for file in helper_templates or []:
            self._write(file.relpath, file.content)

        ctx = FeatureContext(
            problem_name=problem_name,
            task_name=task_name,
            workdir=self.root,
        )
        for feature in features:
            for file in feature.templates(ctx):
                self._write(file.relpath, file.content)

    def solution_path(self) -> str:
        """Absolute path to ``solution.py`` (written by ``ControllerFeature``)."""
        return os.path.abspath(os.path.join(self.root, "solution.py"))

    def _write(self, relpath: str, content: str) -> None:
        path = os.path.join(self.root, relpath)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
