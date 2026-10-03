"""What a code-agent backend can and cannot do.

The loop and the tool-exposure layer branch on this dataclass, never on the
concrete adapter type. New backends declare their capabilities; degradation is
data-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

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
#   "hermes_xml" - same bash-only agent, but taught to emit its one command as a Qwen/hermes
#                  `<tool_call><function=Bash><parameter=command>` call - so an RL-locked model
#                  that emits its native markup regardless is met by a matching prompt + parser.
#   "glm"        - same, taught GLM's native `<tool_call>Bash<arg_key>command</arg_key>
#                  <arg_value>...</arg_value></tool_call>` shape (GLM half-remembers the opener
#                  but improvises the args under a foreign format; its real template stabilizes it).
# The bash_block-family dialects (bash_block, hermes_xml, glm) differ ONLY in the markup used
# for the single bash command; per-served-model selection is `agent.args.tool_protocol`.
ToolProtocol = Literal["native", "client_cli", "bash_block", "hermes_xml", "glm"]
TOOL_PROTOCOLS: tuple[str, ...] = get_args(ToolProtocol)


def uses_control_cli(protocol: ToolProtocol) -> bool:
    """Whether framework tools (SubmitSolution/ExitTask) reach the agent over the workdir control
    CLI (``framework/control.py`` -> the env server's ``/control`` route) instead of as in-process
    loop tools. Every protocol EXCEPT the in-process ``native`` backend does. This is the single
    source of truth two layers must share: the prompt fragment (``builder``) that tells the model
    HOW to call a framework tool, and the control-channel BINDING (``task.py`` -> ``bind_control``)
    that makes that call answerable. If they disagree, an agent is taught to hit a channel that was
    never bound and every submit/exit 503s - the seam bug that hid glm/hermes_xml behind the
    final-eval. Adding a new dialect updates ``ToolProtocol`` only; both layers follow from here."""
    return protocol != "native"


@dataclass(frozen=True)
class Capabilities:
    """Static description of one backend's surface."""

    # How regact's system prompt is applied:
    #   "replace" — we own the whole prompt (Alan custom_system_prompt)
    #   "append"  — we add to the backend's base prompt (Claude --append-system-prompt)
    system_prompt: Literal["replace", "append"]
    tool_protocol: ToolProtocol
    permission_hooks: bool  # supports PreToolUse hooks (path confinement, etc.)
    streams_tool_calls: bool  # surfaces ToolCall events in its stream
    supports_inject: bool  # accepts mid-turn injected messages
    writes_native_transcript: bool  # writes its own session dir (.alan / .claude)
    # The agent EXECUTES its own native tools (Alan, via alancode) — so the loop must NOT also
    # run them (it would double-submit); it only observes. False (default): the loop runs the
    # framework tools when it sees the agent's ToolCall (the scripted test backend relies on this).
    executes_tools: bool = False
