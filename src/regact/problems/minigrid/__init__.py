"""MiniGrid problem package. Importing it registers the problem and re-exports its API."""

from regact.problems.minigrid.problem import MiniGridProblem, MiniGridRenderer
from regact.problems.minigrid.tasks import ALL_MINIGRID_TASKS, LITE_MINIGRID_TASKS

__all__ = [
    "ALL_MINIGRID_TASKS",
    "LITE_MINIGRID_TASKS",
    "MiniGridProblem",
    "MiniGridRenderer",
]
