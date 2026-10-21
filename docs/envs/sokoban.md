# Sokoban (`problem=sokoban`)

Box-pushing puzzles: the player walks on a grid, pushes boxes (never pulls them) and completes a level once every box is on a goal. regact plays packs of [Boxoban](https://github.com/google-deepmind/boxoban-levels) levels (DeepMind, Apache-2.0, licence in `problems/grid_games/data/BOXOBAN_LICENSE`) with a small in-house engine. It is a [grid game](../environments.md#grid-games): ARC's conventions, a colour grid, RESET as action 0, and RHAE against a reference player.

## Why it is in the benchmark

- **The rules are few and exact**, so a correct world model is short (about 30 lines). What a CWM agent buys is planning: once the model is verified, search over it finds the move-optimal solution.
- **Mistakes are irreversible.** A box pushed into a corner can never be recovered (a deadlock), and only RESET escapes it. An agent that predicts the result of a push before playing it avoids paying for a dead end; an agent that acts and observes pays for every deadlock in actions.
- **The efficiency baseline is exact**: each level's reference is its move-optimal solution length, not a human's play, so RHAE measures distance to the optimum.
- No hidden state: the grid shows everything. What the agent cannot know in advance is the layout of the next level, as in ARC.

## Characteristics, and how they were checked

| Property | Value | How it was checked |
|---|---|---|
| Deterministic | yes, no randomness anywhere | the engine draws no random numbers; replaying the stored optimal plans reproduces the same frames (`tests/test_grid_games.py`) |
| Fully observable | yes: grid + level index = state | the engine state is the player cell and the box cells, both drawn |
| Episode length | no step limit, no lose state | `limits.max_actions_*` are regact's own |
| Reset | RESET (action 0) restarts the current level; it is not scored, and there is no framework reset command | as in ARC |
| Reference | move-optimal plan per level | A* with an admissible box-to-goal matching heuristic, every plan replayed through the engine before it is stored |

## Tasks

8 packs, `sokoban_p01` .. `sokoban_p08`, of 7 levels each (10x10 grids, 4 boxes). Each pack mixes 2 `unfiltered`, 3 `medium` and 2 `hard` Boxoban levels, ordered by optimal length:

| Pack | Optimal moves per level | Total |
|---|---|---|
| sokoban_p01 | 31 31 38 42 47 54 56 | 299 |
| sokoban_p02 | 25 30 39 58 64 83 95 | 394 |
| sokoban_p03 | 15 32 50 52 54 54 56 | 313 |
| sokoban_p04 | 26 36 41 49 49 55 60 | 316 |
| sokoban_p05 | 24 24 31 53 60 63 75 | 330 |
| sokoban_p06 | 24 40 47 50 51 66 75 | 353 |
| sokoban_p07 | 25 33 34 45 57 59 61 | 314 |
| sokoban_p08 | 30 33 35 36 36 37 46 | 253 |

`problem.kwargs.max_levels=N` keeps the first N levels of each pack (the shortest), and completing level N wins.

## Observation and actions

`obs.frame` is a one-element list holding the grid. The agent sees colour ids only, never the Sokoban symbols:

| Cell | Colour id |
|---|---|
| floor | 0 |
| wall | 5 |
| goal | 14 |
| box | 12 |
| box on a goal | 9 |
| player | 8 |
| player on a goal | 6 |

Actions: 0 RESET, 1 up, 2 down, 3 left, 4 right. A move into a wall, or a push of a box against a wall or another box, changes nothing (it still counts as an action). `obs.info` holds `state` (`NOT_FINISHED`/`WIN`), `levels_completed` and `win_levels`.

## Use it

The config is [`conf/problem/sokoban.yaml`](../../src/regact/conf/problem/sokoban.yaml). It needs nothing beyond the base install.

```bash
# one pack, the first 3 levels, CWM, driven by Claude Code
make run ARGS="agent=claude problem=sokoban protocol=cwm features=none \
  'problem.tasks=[sokoban_p01]' problem.kwargs.max_levels=3"

# every pack, vanilla
make run ARGS="agent=claude problem=sokoban protocol=vanilla features=none"
```

For a benchmark, copy a local launch script (`scripts/local/bench_regact_*_arc_*.sh`) and change the problem block. `objective_note` should say the reference is an optimal player, not a human. `make viz EXP=<experiment dir>` shows the levels-over-actions curve against the optimal player.

## Rebuild the packs

```bash
git clone --depth 1 https://github.com/google-deepmind/boxoban-levels /tmp/boxoban
python scripts/grid_games/build_sokoban_packs.py /tmp/boxoban --packs 8 --seed 0
```

The build is deterministic for a given seed and takes a few minutes; it rewrites `src/regact/problems/grid_games/data/sokoban_packs.json`.

## Limitations

- Sokoban is famous and its rules are easy to recognise from a few moves. The game is never named to the agent, but a strong model may still identify it.
- Boxoban's `hard` set is hard for model-free RL, not for search: an A* solver finishes every level here in seconds.
