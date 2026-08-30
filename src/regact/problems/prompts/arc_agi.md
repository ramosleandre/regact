# Game: ARC-AGI-3 ({task})

ARC-AGI-3 is an interactive game benchmark. Each game is a multi-level puzzle on a 64x64 grid (cell values 0-15). You must complete all levels to win. The game is deterministic - the same actions reproduce the same outcome - but the rules differ per game and must be discovered through interaction, not assumed.

## Observation

`make_env()` gives you an isolated client with a gym-like interface (importable with `from framework.make_env import make_env`). Each `obs` is:

- `obs.frame`: the current frame(s), a list of 64x64 grids of integer cells (0-15). The last grid is the current state (you can receive multiple frames after an action if something moved during it).
- `obs.available_actions`: the integer action ids currently valid.
- `obs.is_done` / `obs.reward`: episode end / reward (1.0 on WIN).
- `obs.info`: readable metadata: `obs.info["state"]` (`NOT_FINISHED`/`WIN`/`GAME_OVER`), `obs.info["levels_completed"]`, `obs.info["win_levels"]`.

`code_library/arc_agi_helper.py` provides this game's action-id constants and a click builder.

## Goal

You must figure out the goal of the game by yourself. Levels share similar goals. Complete as many levels as possible. As soon as you make progress in terms of levels, submit a solution (`python framework/control.py SubmitSolution`), then keep working on solving the next levels.

## How to read and solve an ARC game

These games are designed to be solved by a human playing them: form a hypothesis about the rules, test it, refine. Do that programmatically, not by eyeballing raw integers:

- Render `obs.frame`'s grid to a PNG (e.g. with matplotlib) and LOOK at the image - objects, walls, a moving token, symmetry and repeated tiles are obvious to the eye but hidden in raw numbers.
- Segment the grid programmatically: flood-fill contiguous same-colour regions. ARC "objects" are usually connected colour blocks; tell them apart from UI strips by size and position.
- Diff consecutive frames (before vs after ONE action) to isolate exactly what that action changed and where the cursor/player is.
- Look for a HUD: a border or top row often encodes a budget, score or level, separate from the play area.
- Learn the action semantics by a systematic single-action sweep - what a click vs a move does, and which actions need a precondition.
- Levels are compositional: they share one rule that ramps in difficulty. Crack level 1's rule, then look for how it generalises or mutates in later levels.
- Ask, like a human: what state counts as WIN, what are the objects, what do the actions do, and what is the constraint (a move budget, a timer)?
