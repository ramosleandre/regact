# Game: MiniGrid ({task})

MiniGrid is a 2D grid world. The agent occupies one cell, faces one of four
directions, and acts by discrete actions. Episodes are stochastic, so a good
policy must generalize across resets — do not hard-code a fixed action sequence.

## Observation

`obs.frame` is the rendered observation (a small grid / image as nested lists,
plus the agent's direction and the mission string when present). Inspect it from
your own scripts with `make_env()` before committing to a policy.

## Actions

`obs.available_actions` lists the valid integer actions. In MiniGrid they are
typically: turn left, turn right, move forward, pick up, drop, toggle, done.
Probe their effect by stepping the env and watching how `obs.frame` changes.

## Goal

Reach the green goal tile (or complete the stated mission) in as few steps as
possible; the episode ends with a positive reward on success.
