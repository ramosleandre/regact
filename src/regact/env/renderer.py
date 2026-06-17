"""Server-side observation rendering: native obs -> agent-facing ``Obs``.

The single place where a native observation becomes what the agent sees
(``Obs.frame`` holds the rendered result; raw extras go in ``Obs.info``). Runs
server-side, so it is the only seam for parsing / VLM-captioning, and it cannot
be supplied by the agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from regact.envclient.obs import Obs


def jsonify(value: Any) -> Any:
    """Make a native observation JSON-safe (numpy arrays/scalars -> nested lists).

    Shared by renderers whose native obs holds numpy data (MiniGrid images, ARC
    frame grids), so it survives the HTTP boundary as plain JSON.
    """
    if hasattr(value, "tolist"):  # numpy array / scalar
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value


class ObsRenderer(ABC):
    """Turn a native observation into an agent-facing ``Obs``."""

    @abstractmethod
    def render(self, native_obs: object, info: dict[str, Any] | None) -> Obs: ...


class RawRenderer(ObsRenderer):
    """Pass the native obs through unchanged; pull ``available_actions`` from info."""

    def render(self, native_obs: object, info: dict[str, Any] | None) -> Obs:
        info = info or {}
        return Obs(
            frame=native_obs,
            available_actions=list(info.get("available_actions", [])),
            info=dict(info),
        )
