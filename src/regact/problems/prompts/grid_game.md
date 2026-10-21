# Game

This is an interactive puzzle game played on a small grid. It has several levels, and you must complete all of them to win. The game is deterministic (the same actions reproduce the same outcome), but its rules and its goal are not given: you must discover them through interaction rather than assume them.

## Observation

{interaction_note}

- `obs.frame`: a one-element list holding the current grid (a list of rows); {frame_note}. Read it as `obs.frame[-1]`. The grid size can change between levels.
- `obs.available_actions`: the integer action ids currently valid.
- `obs.is_done` / `obs.reward`: True / 1.0 once the game is won.
- `obs.info`: readable metadata: `obs.info["state"]` (`NOT_FINISHED`/`WIN`), `obs.info["levels_completed"]`, `obs.info["win_levels"]`.

Action 0 (RESET) restarts the current level from its initial layout. {action_note} `framework/game_helper.py` provides the action-id constants.

## Goal

Discover the goal of the game from your experiments. Levels share related goals. Completing a level loads the next one; completing the last level wins the game.

## General advice

- Understand what each action does before planning long sequences.
- Avoid getting overconfident in a hypothesis: verify it with precise experiments rather than assuming.
- When image viewing is available, look at the grid and try to identify objects and what they represent. Compare the grid before and after an action, or after different actions. Combine this with programmatic analysis to check patterns precisely.
- Choose experiments that distinguish competing explanations or make progress toward the goal.
- Levels share rules that ramp in difficulty. Crack the first level's rule, then look for how it applies in later levels.

{levels_to_win}
