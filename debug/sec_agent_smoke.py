"""Run a code-agent CLI INSIDE the OS sandbox and check it still works.

Run:  make debug D=sec_agent_smoke ARGS=claude       (or ARGS=codex)

Builds the agent + its per-agent deny-default sandbox wrapper exactly like run_task
(allow_read = regact src + the agent's own host dirs; TMPDIR inside the workdir),
sends a one-word prompt, and prints the events. If the agent can't start, the
sandbox is too tight -- the error shows which path to add to its host_read_paths().
"""
import asyncio
import os
import shutil
import sys
import tempfile

from regact.agent.claude_adapter import ClaudeAgent
from regact.agent.codex_adapter import CodexAgent
from regact.security.runtime import detect, make_wrapper


def _src_dir() -> str:
    import regact
    return os.path.dirname(os.path.dirname(os.path.abspath(regact.__file__)))


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "claude"
    if name not in ("claude", "codex") or shutil.which(name) is None:
        print(f"need claude/codex installed + authed; got {name!r}")
        return
    agent = ClaudeAgent() if name == "claude" else CodexAgent()
    workdir = tempfile.mkdtemp(prefix=f"sec-smoke-{name}-")
    agent_tmp = os.path.join(workdir, "tmp")
    os.makedirs(agent_tmp, exist_ok=True)
    src = _src_dir()
    wrap = make_wrapper(detect(), workdir=workdir, allow_read=[src, *agent.host_read_paths()])
    print(f"--- {name} | sandbox={detect().value} | workdir={workdir}")
    print(f"    allow_read = {[src, *agent.host_read_paths()]}")
    await agent.start(
        cwd=workdir, model=None, base_url=None, api_key=None,
        system_prompt="You are a test harness probe. Be terse.",
        env={"PYTHONPATH": src, "TMPDIR": agent_tmp}, runtime_wrap=wrap,
    )
    async for event in agent.send("Reply with a single word: ok."):
        print("  ", type(event).__name__, event)
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
