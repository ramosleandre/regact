# Game: ARC-AGI-3 ({task})

ARC-AGI-3 is an interactive arcade game. Each game is played as a sequence of
levels; you act, observe the resulting frame, and infer the rules from
interaction alone. The game is deterministic, so the same actions reproduce the
same outcome — but rules differ per game and must be discovered, not assumed.

## Observation

`obs.frame` is the current frame as a grid of integer colour cells.
`obs.available_actions` lists the valid actions for this game.

## Goal

Complete as many levels as possible. Progress is measured by levels completed;
the episode advances level by level. Build and consult a model of the game's
dynamics in your `code_library/` rather than guessing.
