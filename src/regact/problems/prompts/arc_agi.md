# Game: ARC-AGI-3 ({task})

ARC-AGI-3 is an interactive game benchmark. Each game is a multi-level puzzle on a 64x64 grid (cell values 0-15). You must complete all levels to win. The game is deterministic - the same actions reproduce the same outcome - but the rules differ per game and must be discovered through interaction, not assumed.

## Observation

{interaction_note}

- {frame_desc}
- `obs.available_actions`: the integer action ids currently valid.
- `obs.is_done` / `obs.reward`: episode end / reward (1.0 on WIN).
- `obs.info`: readable metadata: `obs.info["state"]` (`NOT_FINISHED`/`WIN`/`GAME_OVER`), `obs.info["levels_completed"]`, `obs.info["win_levels"]`.

`framework/arc_agi_helper.py` provides this game's action-id constants and a click builder.

## Goal

{goal_note}

## General advices on how to solve an ARC game

These games are remarkably well solved by humans in comparison to AI. This means that it often requires human-intuitive approaches that over-focused AI may miss. Some advices :

- Try to properly understand the action semantics.
- You may sometime find interest in adopting the perspective of a human playing a small arcade game.
- You should avoid getting overconfident on an hypothesis and stay open to changing your minds to avoid cognitive traps. You should verify your hypothesis with precise experiments and not assume things.
- When image viewing is available, you may find interest in checking the grid visually, and try to identify objects and what they represent. You can compare two images where a different action was played, or the before/after comparison. You can also combine it with programmatic analysis to identify and check patterns precisely.
- Choose experiments that distinguish competing explanations or make progress toward the game's objective.
- Levels are compositional: they share one rule that ramps in difficulty. Crack level 1's rule, then look for how it generalises or mutates in later levels. New levels will have new mechanisms absent in previous one, but the rules are conserved across levels. Recheck your understanding when new observations or levels reveal behavior your current explanation does not account for.

{levels_to_win}
