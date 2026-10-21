"""Level-based grid games; importing registers each game."""

from regact.problems.grid_games import babaisai, puzzlescript, sokoban  # noqa: F401

# the registered problem names, for tools that list every task
GRID_GAMES = ("sokoban", "babaisai", "puzzlescript")
