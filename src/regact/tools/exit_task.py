"""The exit tool.

Signals that the agent is finished. The keep-alive loop checks the flag and
returns control; a final evaluation of the kept controller runs on teardown.
"""

from __future__ import annotations

from typing import Any

from regact.session.state import ExperimentState
from regact.tools.base import Tool, ToolContext, ToolOutput


class ExitTask(Tool):
    """Mark the task as voluntarily finished by the agent."""

    def __init__(self, experiment: ExperimentState) -> None:
        self._experiment = experiment

    @property
    def name(self) -> str:
        return "ExitTask"

    @property
    def description(self) -> str:
        return "Finish the task. Call this once your best controller is submitted and you are done."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def call(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        self._experiment.exit_requested = True
        return ToolOutput(data="Task marked as finished; the run will wrap up.")
