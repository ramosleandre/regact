# Game: ARC-AGI-3 ({task})

ARC-AGI-3 is an interactive game benchmark. Each game is a multi-level puzzle on a
64x64 grid (cell values 0-15). You must complete all levels to win. The game is
deterministic - the same actions reproduce the same outcome - but the rules differ
per game and must be discovered through interaction, not assumed.

## Observation

`make_env()` gives you an isolated client with a gym-like interface (importable with
`from framework.make_env import make_env`). Each `obs` is:

- `obs.frame`: the current frame(s), a list of 64x64 grids of integer cells (0-15). The
  last grid is the current state (you can receive multiple frames after an action if
  something moved during it).
- `obs.available_actions`: the integer action ids currently valid.
- `obs.is_done` / `obs.reward`: episode end / reward (1.0 on WIN).
- `obs.info`: readable metadata: `obs.info["state"]`
  (`NOT_FINISHED`/`WIN`/`GAME_OVER`), `obs.info["levels_completed"]`,
  `obs.info["win_levels"]`.

`code_library/arc_agi_helper.py` provides this game's action-id constants and a click
builder.

## Goal

Complete as many levels as possible (a WIN completes all levels).
