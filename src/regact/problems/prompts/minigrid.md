# Game: MiniGrid ({task})

MiniGrid is a symbolic 2D grid world. Read the current objective from `obs.frame["mission"]`.

This game may or may not be stochastic: layouts and start states can differ between episodes/seeds. If it is, a fixed action sequence will not generalise - your `act(obs)` must react to the current observation, and you should evaluate your controller across several `make_env()` episodes.

`obs.frame` is a dict with keys `image`, `direction`, and `mission`.

{obs_section}

## Actions and interaction

Action IDs: `0 turn left, 1 turn right, 2 move forward, 3 pickup, 4 drop, 5 toggle, 6 done`.

Only action `2` changes the agent's position. Actions `3/4/5` operate on the cell DIRECTLY IN FRONT of the agent - so to pick up a key or toggle a door, navigate to a passable cell ADJACENT to it and FACE it; you can never stand on a key or a closed/locked door.

The agent moves ONTO a goal cell to reach it (goals are entered, not interacted with). Walls, closed/locked doors, and portable objects (keys, balls, boxes) block forward movement; empty cells, floor, open doors, and goals are passable; lava is passable but ends the episode without success.

The agent carries at most one object; a successful pickup removes it from the grid, so remember you are carrying it. Toggling a closed door opens it; a LOCKED door opens only when toggled while carrying a key of the same color. Interaction actions silently do nothing when their preconditions are not met, so do not assume success just because you returned action 3 or 5.

`obs.available_actions` lists accepted IDs, not context-legal ones. Action `6` (done) is task-specific and is a no-op in the standard Empty/DoorKey tasks; do not assume it ends the episode.

## Completion

Follow `obs.frame["mission"]`. Reaching the goal typically ends the episode with a positive reward; stop acting when `obs.is_done` is true. A positive terminal reward means success.
