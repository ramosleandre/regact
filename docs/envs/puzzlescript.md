# PuzzleScript (`problem=puzzlescript`)

[PuzzleScript](https://www.puzzlescript.net) (increpare, MIT) is a language for turn-based grid puzzles: a game declares its objects, collision layers, rewrite rules (`[ > Player | Crate ] -> [ > Player | > Crate ]`) and win conditions (`All Crate on Target`), plus a list of levels. Every game has rules of its own, so each task is a new set of rules to discover. regact plays games from the [puzzlescript.net gallery](https://www.puzzlescript.net/Gallery/) with the original engine, run unmodified under Node. It is a [grid game](../environments.md#grid-games): RESET as action 0, a win after the last level, and ROAE against an optimal player.

## Why it is in the benchmark

- **Many rule sets, each short and exact.** A game is a few to a few dozen rewrite rules, so a correct world model is small, but each game's rules differ (pushing, pulling, growing, beams, gravity, several bodies moved by one input, switching characters). It tests whether a verified world model helps across rule families, not on one game.
- **Planning matters.** Dead ends are common and only RESET escapes them; a model checked against the data lets an agent search before acting.
- **The reference is exact**: each level's shortest solution, found by breadth-first search on the engine.

## Characteristics, and how they were checked

| Property | Value | How it was checked |
|---|---|---|
| Deterministic | yes | the games pass a strict filter: no `random` rule (the engine flags them; on a 952-game corpus every seed-dependent game was flagged), no `realtime`, `again`, `rigid` or `checkpoint`; replaying the same actions in two fresh engines gives the same observations (`tests/test_grid_games.py`) |
| Fully observable | yes | a cell lists every object in it, one per collision layer, so the frame is the whole level state; message screens are skipped and mid-level messages never shown |
| Reset | RESET (action 0) restarts the current level with a fresh compile; it is not scored | as in ARC and the other grid games |
| Undo | not available | the engine's undo is never exposed |
| Reference | shortest solution per level | breadth-first search over full level states (60 s, then 600 s per level), every plan replayed through regact's environment before it is stored |

## Tasks

The benchmark version is **puzzlescript-bench-v1** (`version` in `data/puzzlescript_games.json`): 12 games, 101 levels. Each task is one game with rules of its own; its levels are its first playable levels in the designer's order (at most 10), which teach the mechanics one after the other.

| Task | Game | Mechanic | Levels | Shortest solution per level |
|---|---|---|---|---|
| puzzlescript_01 | Kettle | several bodies, one input | 10 | 4 2 2 4 5 4 6 6 8 10 |
| puzzlescript_02 | Take Heart Lass | spreading despair, blocking | 10 | 3 9 8 8 27 13 17 21 15 15 |
| puzzlescript_03 | Modality | pushing within a floor colour | 4 | 6 33 66 158 |
| puzzlescript_04 | Long Haul Space Flight | player and crates swap on tiles | 10 | 25 11 22 21 29 11 24 10 45 50 |
| puzzlescript_05 | Love and Pieces | touched blocks join the body | 8 | 4 14 9 21 14 21 19 15 |
| puzzlescript_06 | Cake Monsters | all monsters move; colour mixing | 10 | 3 10 26 16 14 24 13 26 13 18 |
| puzzlescript_07 | Stairways | rows of boxes move together | 3 | 14 10 19 |
| puzzlescript_08 | Singleton Traffic | sliding under adjacency rules | 6 | 6 14 10 8 16 14 |
| puzzlescript_09 | Heroes of Sokoban | switching characters (action key) | 10 | 8 20 10 32 20 30 33 22 23 11 |
| puzzlescript_10 | Soliquid | liquid body, freeze and melt | 10 | 6 12 9 16 23 14 20 36 21 48 |
| puzzlescript_11 | Smother | two characters, beams | 10 | 10 19 18 19 30 6 11 9 19 19 |
| puzzlescript_12 | Manic Ammo | remote triggers, guns, mines | 10 | 9 9 10 9 19 11 39 25 5 21 |

A game is kept only when the search solves every one of these levels (60 s, then 600 s): a game is never cut short. Left out: Microban (level 13 unsolved in 600 s), Lime Rick (level 3 unsolved in 600 s), Collapsable Sokoban (level 5 unsolved in 600 s), Midas (level 13 unsolved in 600 s), 2D Whale World (level 13 unsolved in 600 s). Love and Pieces' last level is its ending screen (a block sealed in the wall, no solution exists), so it is not a level.

`problem.kwargs.max_levels=N` keeps the first N levels of each game.

## Observation and actions

`obs.frame` is the grid, a list of rows; each cell is the sorted list of the integer ids of its objects (the background object is in every cell). Actions: 0 RESET, 1 up, 2 down, 3 left, 4 right, and 5, the game's own extra action, for games that have one. An action that changes nothing still counts. Pictures saved for the agent draw the background black and every other object of a cell as a vertical stripe, in id order: ids 1-14 a flat colour, 15-28 the same colour with a black hole, assigned arbitrarily. Every cell content has its own tile, so a picture decodes back to its grid and shows nothing the observation does not; the viewer, videos and icons draw the games' own sprites.

## Use it

Install Node (tested with v22), then fetch the engine and the games:

```bash
make install-puzzlescript
```

The games are fetched, not redistributed: their authors' gists carry no licence. `data/puzzlescript_games.json` pins each source by URL and SHA-256, and the installer refuses a source that changed. They go to `REGACT_PUZZLESCRIPT_DIR` (default `~/.cache/regact/puzzlescript`), outside the agent's sandbox. On a cluster, install on a login node (it needs internet); the runs then need only Node, on the `PATH` of the job (it runs in the trusted environment server, not in the agent's sandbox).

Getting Node:
- macOS: `brew install node`, or the installer from [nodejs.org](https://nodejs.org).
- Linux without a package or module (as on Adastra): the official `linux-x64` tarball; only `bin/node` is needed, e.g. `tar -xJf node-v22.17.0-linux-x64.tar.xz --strip-components=2 node-v22.17.0-linux-x64/bin/node`, then put its directory on `PATH`. Check the tarball against the release's `SHASUMS256.txt`.
- Set `REGACT_PUZZLESCRIPT_DIR` when the home directory is small (a cluster's work space is a better place).

The config is [`conf/problem/puzzlescript.yaml`](../../src/regact/conf/problem/puzzlescript.yaml).

```bash
make run ARGS="agent=claude problem=puzzlescript protocol=cwm features=none \
  'problem.tasks=[puzzlescript_01]' problem.kwargs.max_levels=3"
```

## Rebuild the packs

```bash
python scripts/grid_games/build_puzzlescript_packs.py --budget 60 --retry-budget 600
```

It searches each game's levels in order (60 s, then 600 s for a level not solved), leaves out any game with an unsolved level, replays every plan, and rewrites `data/puzzlescript_games.json`. Search results are cached in `~/.cache/regact/grid_game_solutions/`, so a rebuild only searches new levels. The game list and its pinned sources are at the top of the script; change `VERSION` when the set changes.

## Limitations

- Gallery games are well known and likely in pretraining data; names, titles and messages are hidden, but a model may recognise a mechanic (Microban is plain Sokoban, kept as a control).
- The strict filter removes sliding and animated-gravity games (`again`), so the set leans toward one-step local rewrites.
- Each environment runs a Node process; the engine itself takes well under a millisecond per move.
