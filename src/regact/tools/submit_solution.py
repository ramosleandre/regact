"""The unified submit tool.

One tool for both regimes. It hands the agent's ``solution.py`` to the
``ControllerExecutor``; its behaviour (N episodes vs level boundary) follows
the run's lifecycle. There is no second "SubmitSolutionCompetition" class.
Provided by ``ControllerFeature``. The executor lands in Block 6; it is referenced
by its contract (``run(...) -> EvalResult``) and imported only for typing.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from regact.config.schema import Lifecycle
from regact.controllers.executor import ControllerExecutor
from regact.session.state import ExperimentState
from regact.tools.base import Tool, ToolContext, ToolOutput


class SubmitSolution(Tool):
    """Evaluate the submitted controller and record the result."""

    def __init__(
        self,
        experiment: ExperimentState,
        executor: ControllerExecutor,
        *,
        solution_path: str,
        submissions_dir: str,
        task_name: str,
        lifecycle: Lifecycle,
        n_episodes: int = 1,
        max_moves: int = 400,
    ) -> None:
        self._experiment = experiment
        self._executor = executor
        self._solution_path = solution_path
        self._submissions_dir = submissions_dir
        self._task_name = task_name
        self._lifecycle = lifecycle
        self._n_episodes = n_episodes
        self._max_moves = max_moves

    @property
    def name(self) -> str:
        return "SubmitSolution"

    @property
    def description(self) -> str:
        return (
            "Evaluate the controller in solution.py against the real environment and "
            "record the score. Submit whenever you want to measure your current policy."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def call(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        """Run the executor on ``solution.py``; persist ``submissions/<n>/results.json``;
        bump the submission count; return a digest of the aggregate."""
        index = self._experiment.submission_count
        output_dir = os.path.join(self._submissions_dir, str(index))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "results.json")

        # Run the (synchronous) executor off the event loop: when this tool is
        # invoked from the control server, the executor's sync EnvClient must reach
        # the same server, so the loop has to stay free to answer it.
        result = await asyncio.to_thread(
            self._executor.run,
            task_name=self._task_name,
            solution_path=self._solution_path,
            output_path=output_path,
            lifecycle=self._lifecycle,
            n_episodes=self._n_episodes,
            max_moves=self._max_moves,
        )

        self._experiment.submission_count = index + 1
        self._experiment.last_submission_results = result.to_json()
        return ToolOutput(data={"submission": index, "aggregate": result.aggregate})
