# Crafter (`problem=crafter`)

The open-world survival game of [Hafner (ICLR 2022)](https://arxiv.org/abs/2109.06780), played through the `crafter` package (extra `crafter`). The player gathers resources, builds tools and survives; the game counts 22 achievements, each unlocked once.

## Why it is in the benchmark

- **It is stochastic and partially observable.** Creatures move at random, hidden counters drive hunger, thirst and fatigue, and the player sees a 9x7 window of a 64x64 map. A deterministic CWM cannot represent it; it is the test bed of `protocol=cwm_stochastic`.
- **It has a public LLM reference.** [BALROG](https://balrogai.com/) reports the share of the 22 achievements a step-by-step LLM unlocks in one episode.

## Characteristics, and how they were checked

| Property | Value | How it was checked |
|---|---|---|
| Deterministic | no for the agent; yes given the seed and the actions | the same actions on the same seed give the same frames once `_balance_chunk` iterates in a fixed order (upstream iterates a set) |
| Fully observable | no: map window, no position, hidden need counters | read in `crafter/engine.py` and `objects.py` |
| Episode length | a life ends at death or after 10000 actions (the package's own limit) | `crafter.Env(length=10000)` |
| Reset | the `ResetEnvironment` command starts a new life on the same map | `tests/test_crafter.py` |

## Tasks: "seeded Crafter"

A task is one map: `crafter_s<seed>`, for every seed in `problem.kwargs.seeds` (default 0 to 4). Every life of a task starts at the same place of the same map with an empty inventory. Achievements unlocked in earlier lives stay unlocked, and the task is won when all 22 are.

## Observation and actions

`frame` is a dictionary: `view` (7 rows of 9 names: terrain, creatures, `player` at the centre), `facing`, `inventory` (the four needs and the items), `sleeping`, `daylight`. `info["achievements"]` lists the task's unlocked achievements and `info["life_achievements"]` the current life's. The 17 actions are the game's own, as integers; the agent's prompt names neither the actions nor the achievements.

## Frozen evaluation on held-out maps

When a task ends, its final controller (`final_controller.py`, or without one the controller of the last `RunController` call) plays one life on each seed of `problem.kwargs.eval_seeds` (default 100 to 109), for at most `eval_max_actions` (2000) actions, with the accepted CWM and no agent. This is BALROG's setting: 10 episodes of at most 2000 steps. A frozen CWM that cannot explain an observation rebuilds its State from that observation and the life continues. The result is written to `cwm/evaluation.json`. See `protocols/managed/evaluation.py`.

## Metrics

| Metric | Meaning |
|---|---|
| `achievements` | achievements unlocked over the task's lives (0 to 22) |
| `actions_to_last` | real actions played when the last one was unlocked |
| `first_life_achievements`, `best_life_achievements`, `lives` | achievements within one life; `lives` lists, per life, the task's action count at its end and its achievements |
| `eval_progress` | BALROG's progress: percent of the 22 achievements unlocked per held-out life, averaged |
| `eval_crafter_score` | Hafner's score over the held-out lives: `exp(mean(ln(1 + s_i))) - 1`, `s_i` the percent of lives unlocking achievement `i` |
| `eval_success_rates` | the `s_i` |

Differences from BALROG that remain: its agent gets named actions and a text description of the surroundings; its worlds are unseeded.
