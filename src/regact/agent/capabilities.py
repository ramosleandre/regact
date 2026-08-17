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
    # How the agent invokes tools. Selects BOTH the prompt's tool-invocation fragment AND how
    # framework actions (submit/exit) are wired, so no agent is hardcoded anywhere:
    #   "native"     - in-process Python Tool objects (only the scripted test backend); the loop
    #                  runs the framework tools and the prompt says to call them directly.
    #   "client_cli" - a subprocess CLI agent with native file/bash tools (Claude, Codex) that
    #                  reaches the framework tools via the workdir control CLI
    #                  (`python framework/control.py <Tool>`); the prompt lists those commands.
    #   "bash_block" - a bash-only agent (Alan, Mini-SWE-Agent style) that writes ONE fenced
    #                  ```bash block per turn (alancode's BashBlockFormat extracts + runs it);
    #                  the prompt teaches that convention + shell idioms + the control commands.
    tool_protocol: Literal["native", "client_cli", "bash_block"]
    permission_hooks: bool  # supports PreToolUse hooks (path confinement, etc.)
    streams_tool_calls: bool  # surfaces ToolCall events in its stream
    supports_inject: bool  # accepts mid-turn injected messages
    writes_native_transcript: bool  # writes its own session dir (.alan / .claude)
    # The agent EXECUTES its own native tools (Alan, via alancode) — so the loop must NOT also
    # run them (it would double-submit); it only observes. False (default): the loop runs the
    # framework tools when it sees the agent's ToolCall (the scripted test backend relies on this).
    executes_tools: bool = False
