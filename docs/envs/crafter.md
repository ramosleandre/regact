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
| Episode length | a life ends at death or after `problem.kwargs.life_max_actions` actions (2000, BALROG's limit; the package's own is 10000) | `crafter.Env(length=...)` |
| Reset | the `ResetEnvironment` command starts a new life: same map (seeded) or a new one (unseeded) | `tests/test_crafter.py` |

## Tasks: three settings

A life always starts with an empty inventory and nothing unlocked, and is scored by the achievements unlocked within it. No task ends early: one good life is not a solved game.

| Setting | Tasks | Map of each life | What the run is for |
|---|---|---|---|
| Seeded | `crafter_s<seed>`, `problem.kwargs.seeds` | the same map, at the same starting point | playing one world the agent can return to; a life replays exactly from its actions, so remembering action sequences pays |
| Unseeded | `crafter_u` | a new map every life, from the stream `problem.kwargs.map_stream` (life `i` of stream `k` is map `1000 (k + 1) + i`) | playing while learning the game |
| Controller synthesis | `crafter_c` | the maps of `crafter_u` | producing a program, `final_controller.py`, scored without the agent on new maps |

`crafter_u` and `crafter_c` are one task each: its attempts differ by the stream of maps, so a launcher passes `problem.kwargs.map_stream=<attempt>`. Attempts repeated inside one launch (`n_attempts_per_task`) play the same stream.

## Observation and actions

`frame` is a dictionary: `view` (7 rows of 9 names: terrain, creatures, `player` at the centre), `facing`, `inventory` (the four needs and the items), `sleeping`, `daylight`. `info["achievements"]` lists the current life's achievements. An action is one of the game's 17 action names (`move_left`, `do`, `make_wood_pickaxe`, ...).

The prompt gives what BALROG's gives its agent: the game's name, every action with BALROG's one-line description (recipes included), and the list of the 22 achievements.

## What is measured

The x axis is the task's real actions, or its tool calls (the budget of a run); `resets` in `status.json` gives both at the end of each life.

| | Measure | Setting | Metric |
|---|---|---|---|
| a | play in one world | seeded | per-life achievements: `lives`, first, best, mean, worst |
| b | play while learning the game | unseeded | the same |
| c | the controller the agent leaves, on new maps | synthesis | `eval_progress`, `eval_crafter_score`, `eval_success_rates` |
| d | the CWM the agent leaves, on new maps | synthesis, CWM protocols | `eval_model_explained_rate`, `eval_model_bits_per_step` |

**c: frozen controller.** When a synthesis task ends, `final_controller.py` (without one, the controller of the last `RunController` call) plays one life on each seed of `problem.kwargs.eval_seeds` (100 to 129, shown by no task), for at most `eval_max_actions` (2000) actions, with the accepted CWM and no agent. BALROG's setting is 10 such episodes. A frozen CWM that cannot explain an observation rebuilds its State from that observation alone and the life continues (`regrounded` counts these); when it cannot rebuild one either, the controller keeps the State it had (`ungrounded` counts these steps). A controller of uniformly random actions scores 11.4 % progress there.

**d: frozen CWM alone.** The accepted CWM follows the action lists of `problems/crafter/data/reference_actions.json` on held-out maps 100 to 109: per map, one life of a hand-written controller played to the player's death and one of random actions, about 4,100 steps in all, the same for every run. `eval_model_explained_rate` is the share of these steps it explains. `eval_model_bits_per_step` is their cost, a step it does not explain being charged what a model-free coder pays for it (the coder of the validation report's `no_model` reference); `eval_no_model_bits_per_step` is that coder alone on the same steps, the scale to read it against.

Both are written to `cwm/evaluation.json`; see `protocols/managed/evaluation.py`.

## Metrics

| Metric | Meaning |
|---|---|
| `lives` | per life: the task's action count at its end, its achievements, whether the player died (a reset can abandon a life) |
| `first_life_achievements`, `best_life_achievements` | achievements within the first life, and within the best one |
| `mean_life_achievements`, `worst_life_achievements` | over every life started (died, reached the action limit, or abandoned by a reset) |
| `achievements`, `actions_to_last` | achievements unlocked in at least one life of the task, and the real actions played when the last of them first was |
| `eval_progress` | BALROG's progress: percent of the 22 achievements unlocked per held-out life, averaged |
| `eval_crafter_score` | Hafner's score over the held-out lives: `exp(mean(ln(1 + s_i))) - 1`, `s_i` the percent of lives unlocking achievement `i` |
| `eval_success_rates` | the `s_i` |
| `eval_model_explained_rate`, `eval_model_bits_per_step`, `eval_no_model_bits_per_step` | the frozen CWM alone on the reference action lists |
| `eval_time_limit_stops` | evaluation lives and replays cut by the submitted-code time limit (they then measure speed, not play) |

The viewer draws the achievements over a run (against env actions, tool calls or time) in four readings, `problem.progress_curves`:

| Reading | Value at a point of the run |
|---|---|
| current | the achievements of the life in progress: it falls to 0 when a life starts |
| per life | the achievements the life in progress ends with |
| best so far | the most achievements any one life had reached by then |
| cumulative | the achievements unlocked in at least one life by then |

Differences from BALROG that remain: its agent reads a text description of the nearest objects where ours reads the map window as names; its agent has no learning phase and calls the model at every step, where the synthesis track evaluates a program written after one.
