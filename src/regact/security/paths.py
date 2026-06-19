"""Filesystem helper: is a path inside a given root (e.g. the agent's workdir)?

A pure containment predicate, used when assembling a sandbox's allowed-path set and
when flagging a tool call that names a path outside the workdir. The OS sandbox is
what enforces confinement; this only computes containment.
"""

from __future__ import annotations

import os


def path_within(path: str, root: str) -> bool:
    """True iff ``path`` resolves to ``root`` or somewhere beneath it."""
    root_real = os.path.realpath(root)
    target = os.path.realpath(path if os.path.isabs(path) else os.path.join(root, path))
    return target == root_real or target.startswith(root_real + os.sep)
