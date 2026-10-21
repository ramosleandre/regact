"""Level-based grid games (v8-envs); importing registers each game."""

from regact.problems.grid_games import babaisai, sokoban  # noqa: F401

GRID_GAMES = ("sokoban", "babaisai")  # the registered problem names, for tools that list every task
