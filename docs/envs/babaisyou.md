# Baba Is You (`problem=babaisyou`)

The campaign levels of [Baba Is You](https://hempuli.com/baba/) (Hempuli, 2019), played by regact's own engine. The rules of a level are sentences of word blocks on the board (`BABA IS YOU`, `FLAG IS WIN`, `WALL IS STOP`); pushing the words changes the rules. Unlike [BabaIsAI](babaisai.md), these are the game's real levels, with its full grammar (`AND`, `NOT`, `HAS`, `TEXT`, `LEVEL`) and its properties, and nothing is masked: the difficulty is the reasoning. It is a [grid game](../environments.md#grid-games): RESET as action 0, a win after the last level of a task.

## The engine and how it was checked

No open-source engine plays the real levels without the game itself, so regact has its own (`problems/grid_games/babaisyou/engine.py`), written from the [Baba Is You wiki](https://babaiswiki.fandom.com/wiki/Order_of_Operations)'s order of operations:

- each turn, the YOU objects move (in the game's object order: columns left to right, top to bottom), then the MOVE objects (one that bumps into something turns around and moves back);
- a move into a tile checks OPEN/SHUT (both destroyed), then WEAK (a weak solid object does not block; a weak object that cannot move breaks), then PUSH (chains), then STOP; the border is solid;
- the rules are read again, transformations (`X IS Y`, `X IS TEXT`; `X IS X` protects) apply, and the same-tile interactions resolve in the game's order, each deleting before the next looks: SINK, WEAK, HOT/MELT, DEFEAT, OPEN/SHUT, BONUS, then WIN; HAS leaves its object behind; FLOAT objects only interact with FLOAT objects; an object that is both sides of a pair (YOU and DEFEAT, HOT and MELT, OPEN and SHUT) destroys itself;
- the level counts as being on every tile at once (`LEVEL IS HOT` melts every MELT object, `LEVEL IS WIN` wins at once, on the level's own float layer).

The check: [baba-is-harbor](https://github.com/stared/baba-is-harbor) extracted the levels from the game's files with a solution for each, verified on the real game engine. Every one of those 55 solutions wins in regact's engine, at the same move as in the game (`tests/test_babaisyou.py`; in two levels the stored solution keeps pressing after the win, which the game ignores too). Off those paths the engine is as faithful as the wiki's description: mechanics absent from these levels (conditions such as `ON`/`NEAR`/`FACING`, `TELE`, `SHIFT`, `PULL`, `SWAP`, `MORE`, ...) are not implemented.

## Tasks

The benchmark version is **babaisyou-bench-v1** (`data/babaisyou_levels.json`): one task per campaign stage, its levels in the game's numbering, bonus levels last.

| Task | Stage | Levels | Reference moves per level |
|---|---|---|---|
| babaisyou_00 | map | 12 | 8 64 37 55 37 75 27 80 22 170 117 21 |
| babaisyou_01 | 1. the lake | 15 | 32 149 36 66 20 121 77 104 107 79 73 124 138 97 215 |
| babaisyou_02 | 2. solitary island | 18 | 93 29 29 27 38 72 37 84 84 65 68 117 85 32 97 112 51 80 |
| babaisyou_03 | 3. temple ruins | 10 | 81 47 61 100 157 95 81 82 77 188 |

The reference is the move at which the verified solution wins: a solution made by a person, not a proven optimum, so the scores keep ARC's names (RHAE, LRHAE-Uncapped, mean relative actions) and an agent can beat the reference. `problem.kwargs.max_levels=N` keeps the first N levels of each stage.

## Observation and actions

`obs.frame` is the board, a list of rows; each cell lists what is on the tile, as a player sees it:

- objects by name and facing: `"baba:right"`, `"wall:down"`, `"flag:down"`;
- word blocks in capitals: `"BABA"`, `"IS"`, `"YOU"`;
- `"edge"` for the level border.

`obs.info["active_rules"]` lists the rules in force, sorted (`["BABA IS YOU", "FLAG IS WIN", ...]`), as the game lights the words that form them. Entries within a cell are sorted too: the engine's internal order means nothing to a player. HIDE objects are not shown, as in the game. The informative prompt names the game.

Actions: 0 RESET (restarts the current level, never scored), 1 up, 2 down, 3 left, 4 right, 5 wait (the game's idle key: nothing you control moves, the rest of the level takes its turn). Undo is not available.

The agent's picture writes each entry in its tile (words white on black, objects on a colour of their own, a mark for the facing); the engine package is hidden from the agent's sandbox (`secret_modules`).

## Use it

```bash
make install-babaisyou   # fetch the 55 levels and their solutions (pinned URLs, SHA-256 checked)
make run ARGS="agent=claude problem=babaisyou protocol=cwm features=none \
  'problem.tasks=[babaisyou_00]' problem.kwargs.max_levels=3"
```

The levels are the game's designs: they are fetched at install from baba-is-harbor's pinned commit, never redistributed (`REGACT_BABAISYOU_DIR`, default `~/.cache/regact/babaisyou`). Launch scripts: `scripts/local/bench_regact_{claude,codex}_babaisyou_{cwm,vanilla}.sh`.

## Rebuild

```bash
python scripts/grid_games/build_babaisyou_packs.py [--source <local clone>]
```

It replays every solution through the engine and keeps a level only if it wins; change `VERSION` when the levels change. baba-is-harbor also lists stages 4-9 (99 more levels) but without verified solutions, so they are not included yet.

## Limitations

- The engine covers the mechanics of these 55 levels; later stages need more (conditions, TELE, SHIFT, PULL, ...), each to be checked against the game.
- Contamination: the game and its walkthroughs predate every model's training data. The levels and their solutions are well known.
