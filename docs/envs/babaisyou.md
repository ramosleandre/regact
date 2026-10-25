# Baba Is You (`problem=babaisyou`)

The campaign levels of [Baba Is You](https://hempuli.com/baba/) (Hempuli, 2019), worlds 0 to 9, played by regact's own engine. The rules of a level are sentences of word blocks on the board (`BABA IS YOU`, `FLAG IS WIN`, `WALL IS STOP`); pushing the words changes the rules. Unlike [BabaIsAI](babaisai.md), these are the game's real levels, with its full grammar and its properties, and nothing is masked: the difficulty is the reasoning. It is a [grid game](../environments.md#grid-games): RESET as action 0, a win after the last level of a task.

## The engine

No open-source engine plays the real levels without the game itself, so regact has its own (`problems/grid_games/babaisyou/engine.py`), written from the [Baba Is You wiki](https://babaiswiki.fandom.com/wiki/Order_of_Operations)'s order of operations.

- Grammar: `NOUN IS PROPERTY`, `NOUN IS NOUN` (a transformation), `NOUN HAS NOUN`, `NOUN MAKE NOUN`, with `AND`, `NOT`, the conditions `ON`, `NEAR`, `FACING`, and the nouns `TEXT`, `LEVEL`, `EMPTY` (every free tile), `ALL` and `GROUP`; objects that are `WORD` count as their own word. A sentence never starts inside the subject or conditions of another (`KEKE ON GRASS IS DEFEAT` does not also say `GRASS IS DEFEAT`).
- A turn: the YOU objects move (in the game's object order; a mover blocked by another mover tries again once that one has moved, so a line of movers moves as one), then MOVE objects (turning around once when blocked), then SHIFT carries what stands on a belt; the rules are read and transformations apply once (`X IS X` protects; a rule that a transformation spells acts only next turn, as in the game since 2020); then TELE, the SHIFT turn, FALL, the directions (`UP`, ...), and the same-tile interactions in the game's order, each deleting before the next looks: MORE, SINK, WEAK, HOT/MELT, DEFEAT, OPEN/SHUT, MAKE, BONUS, then WIN.
- A move into a tile checks OPEN/SHUT (both destroyed), then SWAP, then WEAK, then PUSH (chains; a stack moves together), then STOP and PULL (solid); what the mover leaves behind is pulled if PULL. FLOAT objects only interact with FLOAT objects; HAS leaves its object behind; the level counts as being on every tile at once (`LEVEL IS WIN`, `LEVEL IS HOT`, ...).
- `EMPTY`'s properties act through a temporary object on every free tile during movement (`EMPTY IS YOU` moves the free tiles, `EMPTY IS PUSH` makes them solid). With three or more teleporters of one kind the game picks the destination at random; the engine draws it from a generator seeded by the level and the turn.

## How it was checked

1. **Real-game solutions (worlds 0-3, 55 levels).** [baba-is-harbor](https://github.com/stared/baba-is-harbor) extracted the levels from the game's files with a solution for each, verified on the real game. Every one wins in regact's engine at the same move as in the game.
2. **Rules at the start (all levels).** For 129 of the 131 levels whose wiki page lists their initial rules, the engine reads the same rules (up to the names harbor gives a few reskinned objects, e.g. the wiki's ROBOT is `statue` here). The two others: a wiki omission (Jump) and Ghost Guard, which harbor's extraction corrupted (excluded).
3. **The wiki's solutions (worlds 4-9, 98 levels).** For each level, an agent followed the wiki's solution in this engine until it won, and reported any behaviour the real game would not show. Each confirmed report became a fix and a test (the condition parse, rows and stacks of movers, pull inside a push chain, EMPTY's properties, ALL, the timing of transformations, ...), re-checked against the 55 real-game solutions; the wiki's own move list for Babas Are You reproduces step by step. These solutions win here; that they would win in the game is checked only as far as the wiki's prose goes.

`tests/test_babaisyou.py` replays every level's solution and covers each mechanic on a small board.

## Tasks

The benchmark version is **babaisyou-bench-v2** (`data/babaisyou_levels.json`), 153 levels: one task per world, its levels in the game's numbering, bonus and extra levels last.

| Task | World | Levels | Main / bonus / extra | Median reference moves | Reference from |
|---|---|---|---|---|---|
| babaisyou_00 | map | 12 | 11/1/0 | 46 | real game |
| babaisyou_01 | the lake | 15 | 13/2/0 | 97 | real game |
| babaisyou_02 | solitary island | 18 | 12/6/0 | 70 | real game |
| babaisyou_03 | temple ruins | 10 | 9/1/0 | 82 | real game |
| babaisyou_04 | forest of fall | 19 | 11/3/5 | 119 | wiki, this engine |
| babaisyou_05 | deep forest | 21 | 14/2/5 | 160 | wiki, this engine |
| babaisyou_06 | rocket trip | 15 | 13/2/0 | 146 | wiki, this engine |
| babaisyou_07 | flower garden | 12 | 10/2/0 | 188 | wiki, this engine |
| babaisyou_08 | chasm | 16 | 0/7/9 | 170 | wiki, this engine |
| babaisyou_09 | volcanic cavern | 15 | 14/1/0 | 189 | wiki, this engine |

A level's reference is the move at which its solution wins: a person's solution for worlds 0-3, an agent's for worlds 4-9 (about twice as long), never a proven optimum, so the scores keep ARC's names (RHAE, LRHAE-Uncapped, mean relative actions) and an agent can beat the reference. `problem.kwargs.max_levels=N` keeps the first N levels of each world.

## Observation and actions

`obs.frame` is the board, a list of rows; each cell lists what is on the tile, as a player sees it:

- objects by name and facing: `"baba:right"`, `"wall:down"`, `"flag:down"`;
- word blocks in capitals: `"BABA"`, `"IS"`, `"YOU"`;
- `"edge"` for the level border.

`obs.info["active_rules"]` lists the rules in force, sorted (`["BABA IS YOU", "FLAG IS WIN", "KEKE ON GRASS IS DEFEAT", ...]`), as the game lights the words that form them. Entries within a cell are sorted too: the engine's internal order means nothing to a player. HIDE objects are not shown, as in the game. The informative prompt names the game.

Actions: 0 RESET (restarts the current level, never scored), 1 up, 2 down, 3 left, 4 right, 5 wait (the game's idle key: nothing you control moves, the rest of the level takes its turn). Undo is not available.

The agent's picture writes each entry in its tile (words white on black, objects on a colour of their own, a mark for the facing); the engine package, which also holds the worlds 4-9 solutions, is hidden from the agent's sandbox (`secret_modules`).

## Use it

```bash
make install-babaisyou   # fetch the 153 levels and the real-game solutions (pinned URLs, SHA-256 checked)
make run ARGS="agent=claude problem=babaisyou protocol=cwm features=none \
  'problem.tasks=[babaisyou_00]' problem.kwargs.max_levels=3"
```

The levels are the game's designs: they are fetched at install from baba-is-harbor's pinned commit, never redistributed (`REGACT_BABAISYOU_DIR`, default `~/.cache/regact/babaisyou`). Launch scripts: `scripts/local/bench_regact_{claude,codex}_babaisyou_{cwm,vanilla}.sh`.

## Rebuild

```bash
python scripts/grid_games/build_babaisyou_packs.py [--source <local clone>]
```

It replays every solution through the engine and keeps a level only if it wins; change `VERSION` when the levels change.

## Limitations

- Worlds 4-9 are checked against the wiki's descriptions, not the game: a detail the wiki does not state (e.g. a MOVE object at the edge of a belt, the exact order of empty tiles moving) may differ from the game.
- Ghost Guard (world 4) is excluded: its file lost three words in extraction.
- Three or more teleporters of one kind make the game random; the engine's draw is reproducible but cannot match the game's.
- Contamination: the game and its walkthroughs predate every model's training data. The levels and their solutions are well known.
