"""Block 4.5 smoke: spawn a real code-agent CLI headless and print its events.

Run:  make debug D=block4_5_agent_smoke ARGS=claude    (or ARGS=codex)
      PYTHONPATH=src python debug/block4_5_agent_smoke.py claude

Checks the CLI is installed + authenticated and that our adapter parses its
stream into normalized AgentEvents. Needs the CLI (see docs/agents-setup.md).
"""

import asyncio
import shutil
import sys
import tempfile

from regact.agent.claude_adapter import ClaudeAgent
from regact.agent.codex_adapter import CodexAgent

_BINARY = {"claude": "claude", "codex": "codex"}


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "claude"
    binary = _BINARY.get(name)
    if binary is None:
        print(f"unknown agent {name!r}; use 'claude' or 'codex'")
        return
    if shutil.which(binary) is None:
        print(
            f"'{binary}' not found on PATH — install + authenticate it first "
            f"(see docs/agents-setup.md)"
        )
        return

    agent = ClaudeAgent() if name == "claude" else CodexAgent()
    workdir = tempfile.mkdtemp(prefix=f"regact-{name}-")
    await agent.start(
        cwd=workdir,
        model=None,
        base_url=None,
        api_key=None,
        system_prompt="You are a test harness probe. Be terse.",
        env={"PYTHONPATH": "src"},
    )
    print(f"--- {name} in {workdir} ---")
    async for event in agent.send("Reply with a single word: ok."):
        print("  ", type(event).__name__, event)
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
