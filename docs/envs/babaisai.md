# BabaIsAI (`problem=babaisai`)

[BabaIsAI](https://github.com/nacloos/baba-is-ai) (MIT, [paper](https://arxiv.org/abs/2407.13729)) is a small grid remake of *Baba Is You*: the rules of a level are sentences of word blocks on the board (`BABA IS YOU`, `DOOR IS WIN`, `WALL IS STOP`), and the player changes the rules by pushing the words. A level is won by moving the object that is YOU onto an object that is WIN. It is a [grid game](../environments.md#grid-games): ARC's conventions, RESET as action 0, and ROAE against an optimal player.

**Status:** kept in the code, but not part of the benchmark. The levels are small (one 8x8 room or two rooms in 13x9), and in a pilot Sonnet 5.5 solved every pack and named the game's mechanism within a few tool calls, so it adds little beyond PuzzleScript.

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
| Reference | shortest solution per level | breadth-first search on the real engine (300 s per level; candidates not solved are cut), every plan replayed through regact's environment before it is stored |

The engine's facing direction is not shown; it affects nothing in these levels.

## Tasks

The benchmark version is **babaisai-bench-v1** (`version` in `data/babaisai_packs.json`): 8 packs, `babaisai_p01` .. `babaisai_p08`, of 8 levels each. Candidates are drawn in turn from 7 skill families (the engine's 43 generators, grouped: go to the win object in one or two rooms, break a STOP rule first, make a WIN rule in one or two rooms, break STOP then make WIN, make another object YOU; the one-move `you_win` generators are left out). A candidate is cut when the search does not solve it within 300 s, when it is solved in fewer than 3 moves, or when an object starts hidden under another. The first 64 levels kept are sorted by shortest solution and dealt in snake order, so every pack has the same easy-to-hard spread (93-97 moves per pack); each pack is ordered by length. The packs are independent replicates of one game.

| Pack | Shortest solution per level | Total |
|---|---|---|
| babaisai_p01 | 3 7 8 13 13 15 15 23 | 97 |
| babaisai_p02 | 3 7 8 12 13 15 16 23 | 97 |
| babaisai_p03 | 3 7 8 12 13 15 16 23 | 97 |
| babaisai_p04 | 3 6 8 11 13 15 17 21 | 94 |
| babaisai_p05 | 3 5 9 11 13 14 17 21 | 93 |
| babaisai_p06 | 3 5 9 11 13 14 17 21 | 93 |
| babaisai_p07 | 4 5 10 11 13 14 18 21 | 96 |
| babaisai_p08 | 5 5 10 10 13 14 18 19 | 94 |

What the selection cut, per family (from `selection` in the data file). Exact references are out of reach for most two-room "make WIN" levels, so those families are under-represented: this is the cost of requiring a proven-optimal reference.

| Skill family | Drawn | Kept | Too short (< 3 moves) | Unsolved in 300 s | Object hidden at start |
|---|---|---|---|---|---|
| goto_win | 13 | 8 | 5 | 0 | 0 |
| make_win | 13 | 13 | 0 | 0 | 0 |
| two_room-goto_win | 13 | 9 | 4 | 0 | 0 |
| two_room-break_stop-goto_win | 13 | 13 | 0 | 0 | 0 |
| two_room-make_you | 12 | 12 | 0 | 0 | 0 |
| two_room-make_win | 12 | 4 | 0 | 7 | 1 |
| two_room-break_stop-make_win | 12 | 5 | 0 | 7 | 0 |

`problem.kwargs.max_levels=N` keeps the first N levels of each pack (the shortest), and completing level N wins.

## Observation and actions

`obs.frame` is the grid, a list of rows. Each cell holds one integer per kind of object or word block on top of it (the same kind always has the same integer); the table is in `data/babaisai_packs.json` and the agent never sees it. Code 0 is an empty cell. The agent's picture gives each code its own tile and no meaning: codes 1-14 are flat colours assigned arbitrarily (no object keeps its engine colour), codes 15-28 the same colours with a black hole, so a picture always decodes back to its grid. The viewer's Human view uses the engine's own tiles.

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
python scripts/grid_games/build_babaisai_packs.py --seed 0 --budget 300
```

It takes about two hours the first time (each candidate cut at the budget costs 300 s); search results are cached in `~/.cache/regact/grid_game_solutions/`, so a rebuild only searches new candidates. It rewrites `src/regact/problems/grid_games/data/babaisai_packs.json` (levels, plans, the selection counts and the code table); change `VERSION` in the script when the levels change.

## Limitations

- The levels come from a handful of generators, so the same few mechanics recur across packs; the original game's hand-made levels are not included.
- The game is a known remake. The prompt never names it and the agent sees integer codes, not words, but a model that recognises the layout may guess the grammar instead of discovering it.
