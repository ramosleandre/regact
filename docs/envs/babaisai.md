# BabaIsAI (`problem=babaisai`)

[BabaIsAI](https://github.com/nacloos/baba-is-ai) (MIT, [paper](https://arxiv.org/abs/2407.13729)) is a small grid remake of *Baba Is You*: the rules of a level are sentences of word blocks on the board (`BABA IS YOU`, `DOOR IS WIN`, `WALL IS STOP`), and the player changes the rules by pushing the words. A level is won by moving the object that is YOU onto an object that is WIN. It is a [grid game](../environments.md#grid-games): ARC's conventions, RESET as action 0, and ROAE against an optimal player.

## Why it is in the benchmark

- **The rules are part of the state.** Pushing a word can make another object the player, remove a wall's STOP, or make a new object WIN. A world model must parse the sentences on the board after every move; a wrong guess about the grammar shows up as a prediction mismatch at once.
- **Discovering the rules is the hard part.** Once the rules are known, a level is short (3 to about 30 moves) and a search over a correct model finds the optimum. That is the case where an enforced, verified world model should help most, and where a model that acts on assumptions pays in wasted moves.
- **The reference is exact**: each level's shortest solution, found by breadth-first search on the real engine.

## Characteristics, and how they were checked

| Property | Value | How it was checked |
|---|---|---|
| Deterministic | yes | a level is (engine id, seed): its layout is the same whatever numpy's and Python's global generators hold, and no move draws randomness (any draw from the engine's generator raises) - `tests/test_grid_games.py` |
| Level layout | procedural, pinned per level | each level stores its engine id and seed |
| Step limit | lifted | the engine's 100-step limit is set to 10^9, so a level ends only when it is won; the step count is never observable |
| Hidden state | objects the player stands on | only the top object of each cell is shown; levels with an object hidden under another at reset are excluded, so whatever is hidden was seen before |
| Reset | RESET (action 0) restarts the current level; it is not scored | as in ARC and the other grid games |
| Reference | shortest solution per level | breadth-first search on the real engine (20 s budget per level; unsolved levels are skipped), every plan replayed through regact's environment before it is stored |

The engine's facing direction is not shown; it affects nothing in these levels.

## Tasks

8 packs, `babaisai_p01` .. `babaisai_p08`, of 7 levels each: 3 single-room levels (8x8) and 4 two-room levels (13x9), drawn from the engine's 43 level generators (go to the win object, break a STOP rule, make a WIN rule, make another object YOU, with distractor objects and rules), ordered by shortest solution. Levels solved in fewer than 3 moves are excluded.

| Pack | Shortest solution per level | Total |
|---|---|---|
| babaisai_p01 | 5 11 12 14 15 20 23 | 100 |
| babaisai_p02 | 3 4 9 12 15 15 16 | 74 |
| babaisai_p03 | 3 4 5 10 10 12 16 | 60 |
| babaisai_p04 | 3 10 11 12 13 13 27 | 89 |
| babaisai_p05 | 3 3 4 8 10 13 14 | 55 |
| babaisai_p06 | 4 6 8 9 10 19 32 | 88 |
| babaisai_p07 | 3 5 8 9 10 13 14 | 62 |
| babaisai_p08 | 4 6 9 10 11 13 22 | 75 |

`problem.kwargs.max_levels=N` keeps the first N levels of each pack (the shortest), and completing level N wins.

## Observation and actions

`obs.frame` is a one-element list holding the grid. Each cell holds one integer per kind of object or word block on top of it (the same kind always has the same integer); the table is in `data/babaisai_packs.json` and the agent never sees it. Code 0 is an empty cell. The video draws word blocks framed, so they read as blocks.

Actions: 0 RESET, 1 up, 2 down, 3 left, 4 right. A move into a STOP object, or a push of a word against one, changes nothing (it still counts as an action).

## Use it

The config is [`conf/problem/babaisai.yaml`](../../src/regact/conf/problem/babaisai.yaml). Install the engine with `make install-babaisai` (pinned commit `40ae1b2`; it pulls the old `gym` 0.25, which prints deprecation warnings). The engine is in the agent's `secret_modules`, so the sandbox hides it.

```bash
# one pack, the first 3 levels, CWM
make run ARGS="agent=claude problem=babaisai protocol=cwm features=none \
  'problem.tasks=[babaisai_p01]' problem.kwargs.max_levels=3"
```

For a benchmark, copy a local launch script and change the problem block; `objective_note` should say the reference is an optimal player.

## Rebuild the packs

```bash
python scripts/grid_games/build_babaisai_packs.py --packs 8 --seed 0
```

It takes about 10 minutes and rewrites `src/regact/problems/grid_games/data/babaisai_packs.json` (levels, plans and the code table).

## Limitations

- The levels come from a handful of generators, so the same few mechanics recur across packs; the original game's hand-made levels are not included.
- The game is a known remake. The prompt never names it and the agent sees integer codes, not words, but a model that recognises the layout may guess the grammar instead of discovering it.
