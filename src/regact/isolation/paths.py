"""Filesystem confinement: is a path inside the agent's workdir?

The structural complement to the backend's native path hook — used to validate
that file access stays within the workdir (the agent must not climb out to read
the game data or framework source).
"""

from __future__ import annotations

import os


def path_within(path: str, root: str) -> bool:
    """True iff ``path`` resolves to ``root`` or somewhere beneath it."""
    root_real = os.path.realpath(root)
    target = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
    return target == root_real or target.startswith(root_real + os.sep)
