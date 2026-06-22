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

The game is created once and is already started, so this returns the same handle
without resetting. Read the current state with ``env.current()``; ``env.reset()``
resets the current level and counts as an action. You never import the game.
"""

from regact.envclient.client import EnvClient

_BASE_URL = {base_url!r}
_GAME_ID = {game_id!r}
_HANDLE = None


def make_env(record_frames: bool = False):
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = EnvClient.connect(_BASE_URL, _GAME_ID)
    return _HANDLE
'''


# A tiny workdir CLI a subprocess agent runs to invoke a framework tool over the
# control channel, e.g. ``python framework/control.py SubmitSolution``. Native
# agents (Alan) don't use it; it is harmless to ship for them. URL/game baked in
# via replace (the body has literal braces, so str.format is unsafe here).
_CONTROL_CLI = '''\
"""Invoke a framework tool over the control channel (e.g. SubmitSolution, ExitTask).

Usage:  python framework/control.py <ToolName> ['<json-input>']
"""

import json
import sys

import httpx

_URL = "__BASE_URL__/control/__GAME_ID__/tool"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    response = httpx.post(_URL, json={"name": name, "input": payload}, timeout=300.0)
    response.raise_for_status()
    print(json.dumps(response.json()))


if __name__ == "__main__":
    main()
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
        self._write(
            "framework/control.py",
            _CONTROL_CLI.replace("__BASE_URL__", env_base_url).replace("__GAME_ID__", game_id),
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
