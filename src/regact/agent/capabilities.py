"""What a code-agent backend can and cannot do.

The loop and the tool-exposure layer branch on this dataclass, never on the
concrete adapter type. New backends declare their capabilities; degradation is
data-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Capabilities:
    """Static description of one backend's surface."""

    # How regact's system prompt is applied:
    #   "replace" — we own the whole prompt (Alan custom_system_prompt)
    #   "append"  — we add to the backend's base prompt (Claude --append-system-prompt)
    system_prompt: Literal["replace", "append"]
    # How framework actions (submit/exit) reach the agent:
    #   "native_tools" — in-process Python Tool objects (only an in-process agent, i.e. Alan)
    #   "client_cli"   — the agent calls them via the workdir client CLI over HTTP
    #                    (any subprocess CLI agent, e.g. Claude Code)
    control_actions: Literal["native_tools", "client_cli"]
    permission_hooks: bool  # supports PreToolUse hooks (path confinement, etc.)
    streams_tool_calls: bool  # surfaces ToolCall events in its stream
    supports_inject: bool  # accepts mid-turn injected messages
    writes_native_transcript: bool  # writes its own session dir (.alan / .claude)
