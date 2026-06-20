"""Block 4 manual smoke: drive a ScriptedAgent through the CodeAgent interface.

Run:  python debug/block4_agent.py
"""

import asyncio

from regact.agent.events import TextDelta, ToolCall, TurnComplete
from regact.agent.scripted_agent import ScriptedAgent


async def main() -> None:
    agent = ScriptedAgent(
        [
            [
                TextDelta("I'll write a controller."),
                ToolCall("c1", "submit_solution", {"path": "ctrl.py"}),
            ],
            [TextDelta("Submitting again."), TurnComplete("done")],
        ]
    )
    await agent.start(
        cwd="/tmp/regact-demo", model=None, base_url=None, api_key=None, system_prompt="sys"
    )
    print("capabilities:", agent.capabilities())

    for turn, msg in enumerate(["start the task", "continue"], start=1):
        print(f"--- turn {turn}: send({msg!r}) ---")
        async for event in agent.send(msg):
            print("  ", type(event).__name__, event)
    await agent.close()
    print("closed:", agent.closed)


if __name__ == "__main__":
    asyncio.run(main())
