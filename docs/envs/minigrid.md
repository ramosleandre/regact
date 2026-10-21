# MiniGrid and BabyAI for CWM (`problem=minigrid_cwm`)

[MiniGrid](https://minigrid.farama.org/) is a family of small symbolic grid worlds (Farama, Apache-2.0): an agent with a facing direction moves, turns, picks up, drops and toggles objects (keys, doors, boxes, balls) to fulfil a mission given as text. [BabyAI](https://minigrid.farama.org/environments/babyai/) adds levels with compositional language missions ("open the red door, then pick up the blue box"). Both are registered by the `minigrid` package and share one observation and action space.

`problem=minigrid` plays any of them; `conf/problem/minigrid_cwm.yaml` is a set of 10 tasks chosen for CWM experiments.

## Why it is in the benchmark

- **Different mechanics from ARC and the grid games**: a facing direction, an inventory of one object, doors with locked/closed/open states, objects that block, lava that ends the episode, and language missions to parse. A world model must get all of them right to replay the data.
- **A text mission**: the goal is stated, unlike ARC, so the difficulty moves from guessing the goal to modelling the dynamics and planning a long sequence (up to ~45 moves).
- **Traps a model can predict**: in RedBlueDoors, opening the blue door before the red one fails the episode; in Fetch-like missions the wrong pickup ends it; lava kills. A verified world model can check a move before it is played.

## Characteristics, and how they were checked

All 178 registered ids (82 `MiniGrid-*`, 96 `BabyAI-*`, minigrid 3.1.0) were checked:

- **Randomness within an episode.** Each env was reset with a seed, its random generator was then replaced by a different one, and the same 200 actions were replayed. A step that draws randomness diverges. Only the 6 `Dynamic-Obstacles` envs diverged (their obstacles move at random on every step); every other id drew no randomness in any step (0 of 92,904 steps checked). All randomness happens at reset and is visible in the first observation.
- **Fixed layout.** With a fixed `problem.seed`, every reset rebuilds the same layout, as an ARC level does.
- **What the full grid hides.** With `fully_obs: true` the whole grid is visible, but MiniGrid does not draw the carried object; regact adds it to the frame as `obs.frame["carrying"]`. The cell under the agent was visible before it entered, the step count is the number of steps since the reset, and the step limit is constant per id (except 9 BabyAI levels where it depends on the mission). The multi-box `ObstructedMaze` variants hide a key inside each box and are excluded: no rule tells which key a box holds.
- **Exact rewards.** The success reward is `1 - 0.9 * step_count / max_steps`, a float a world model must reproduce exactly; regact rounds it to 6 decimals so any correct formula matches.

The determinism check runs in the test suite on the 10 tasks, with Dynamic-Obstacles as a control that must fail it (`tests/test_minigrid_cwm.py`). The CWM protocol cannot detect in-episode randomness by itself (with a fixed seed it would only see transitions it cannot explain), so the task list is what keeps random envs out.

## Tasks

| Task | Domain / trap | Optimal moves (seed 0) | Step limit |
|---|---|---|---|
| MiniGrid-LavaCrossingS9N2-v0 | lava maze, lava ends the episode | 14 | 324 |
| MiniGrid-DoorKey-8x8-v0 | key, locked door, goal | 17 | 640 |
| MiniGrid-RedBlueDoors-8x8-v0 | order rule: the blue door first fails | 19 | 1280 |
| MiniGrid-UnlockPickup-v0 | unlock, drop the key, pick up the box | 16 | 288 |
| MiniGrid-BlockedUnlockPickup-v0 | move a blocking ball first | 24 | 576 |
| MiniGrid-KeyCorridorS4R3-v0 | search side rooms for the key | 20 | 480 |
| MiniGrid-MultiRoom-N6-v0 | long navigation through six doors | 46 | 120 |
| MiniGrid-LockedRoom-v0 | long mission over six rooms | about 40 (36-44 on seeds 0-2) | 190 |
| BabyAI-PutNextLocal-v0 | language: put an object next to another | 6 | 128 |
| BabyAI-OpenDoorsOrderN4-v0 | language: open doors in the stated order | not computed | 720 |

Optimal moves come from a breadth-first search on the real env. The search cannot restore BabyAI's sequence verifier, so OpenDoorsOrder has no figure.

## Observation and actions

`obs.frame` is a dict: `image` (the grid, `[x][y]` of `[object, colour, state]`), `direction` (0 right, 1 down, 2 left, 3 up), `mission` (text) and `carrying` (`[object, colour, state]` or `None`). The prompt (`problems/prompts/minigrid.md` and `minigrid_full.md`) gives the encodings. Actions: 0 turn left, 1 turn right, 2 forward, 3 pick up, 4 drop, 5 toggle, 6 done. The episode ends on success (reward > 0), on a failure (lava, a wrong door or object) or at the step limit.

## Use it

The config is [`conf/problem/minigrid_cwm.yaml`](../../src/regact/conf/problem/minigrid_cwm.yaml): the 10 tasks, `single_instance`, `seed: 0`, `fully_obs: true`, `info_mode: informative`. Install with `make install-minigrid`.

```bash
# one task, CWM
make run ARGS="agent=claude problem=minigrid_cwm protocol=cwm features=none \
  'problem.tasks=[MiniGrid-DoorKey-8x8-v0]'"

# the whole set, vanilla
make run ARGS="agent=claude problem=minigrid_cwm protocol=vanilla features=none"
```

`problem.seed=null` gives a new layout on every reset instead, so the agent must handle any layout. The metrics are `success_rate` and the episode reward; MiniGrid has no levels and no reference player, so it has no RHAE.

## Limitations

- The prompt names MiniGrid and describes its encodings; the task id is not shown, but the mission text often says what the task is.
- The full grid makes MiniGrid's `Memory` envs trivial (the cue is always visible), so none is in the set.
- MiniGrid's `max_steps` is not in the observation: a model learns it only from a success reward or a truncation.
