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
| Reset | the `ResetEnvironment` command starts a new life: same map (seeded) or a new one (unseeded) | `tests/test_crafter.py` |

## Tasks: two settings

Both settings share everything but the map of each life: a life starts with an empty inventory, and the achievements unlocked in earlier lives of the task stay unlocked.

| Setting | Tasks | Map of each life | What it measures |
|---|---|---|---|
| Seeded | `crafter_s<seed>`, `problem.kwargs.seeds` (0 to 4) | the same map, at the same starting point | how far an agent gets in one world it can return to; a life replays exactly from its actions, so remembering action sequences pays |
| Unseeded | `crafter_u<k>`, `problem.kwargs.unseeded` (0 to 4) | a new map every life (life `i` is map `1000 (k + 1) + i`) | what an agent learns about the game: the "training" setting |

A seeded task is won when all 22 achievements are unlocked over its lives. An unseeded task is told that its main result is the final evaluation below.

## Observation and actions

`frame` is a dictionary: `view` (7 rows of 9 names: terrain, creatures, `player` at the centre), `facing`, `inventory` (the four needs and the items), `sleeping`, `daylight`. `info["achievements"]` lists the task's unlocked achievements and `info["life_achievements"]` the current life's. The 17 actions are the game's own, as integers; the agent's prompt names neither the actions nor the achievements.

## What is measured

The x axis is the task's real actions, or its tool calls (the budget of a run).

| | Measure | Setting | Metric |
|---|---|---|---|
| a | play in one world | seeded | `achievements` over the task, with the action and tool-call count at each new one; `lives` |
| b | play while learning the game | unseeded | `lives`: achievements within each life against the actions played so far; `first_life_achievements` is the closest to a BALROG episode |
| c | the controller the agent leaves, on new maps | unseeded | `eval_progress`, `eval_crafter_score`, `eval_success_rates` |
| d | the CWM the agent leaves, on new maps | unseeded, CWM protocols | `eval_model_explained_rate`, `eval_model_bits_per_step` |

**c: frozen controller.** When an unseeded task ends, `final_controller.py` (without one, the controller of the last `RunController` call) plays one life on each seed of `problem.kwargs.eval_seeds` (100 to 109, shown by no task), for at most `eval_max_actions` (2000) actions, with the accepted CWM and no agent. This is BALROG's setting: 10 episodes of at most 2000 steps. A frozen CWM that cannot explain an observation rebuilds its State from that observation alone and the life continues (`regrounded` counts these).

**d: frozen CWM alone.** The accepted CWM follows the action lists of `problems/crafter/data/reference_actions.json` on the same held-out maps: per map, one life of a hand-written controller played to the player's death and one of random actions, about 4,100 steps in all, the same for every run. `eval_model_explained_rate` is the share of these steps it explains. `eval_model_bits_per_step` is their cost, a step it does not explain being charged what a model-free coder pays for it (the coder of the validation report's `no_model` reference); `eval_no_model_bits_per_step` is that coder alone on the same steps, the scale to read it against.

Both are written to `cwm/evaluation.json`; see `protocols/managed/evaluation.py`. A seeded task has neither: its agent learned one map, and other maps would be another distribution.

## Metrics

| Metric | Meaning |
|---|---|
| `achievements` | achievements unlocked over the task's lives (0 to 22) |
| `actions_to_last` | real actions played when the last one was unlocked |
| `first_life_achievements`, `best_life_achievements`, `lives` | achievements within one life; `lives` lists, per life, the task's action count at its end and its achievements |
| `eval_progress` | BALROG's progress: percent of the 22 achievements unlocked per held-out life, averaged |
| `eval_crafter_score` | Hafner's score over the held-out lives: `exp(mean(ln(1 + s_i))) - 1`, `s_i` the percent of lives unlocking achievement `i` |
| `eval_success_rates` | the `s_i` |
| `eval_model_explained_rate`, `eval_model_bits_per_step`, `eval_no_model_bits_per_step` | the frozen CWM alone on the reference action lists |
| `eval_time_limit_stops` | evaluation lives and replays cut by the submitted-code time limit (they then measure speed, not play) |

Differences from BALROG that remain: its agent gets named actions and a text description of the surroundings; its worlds are unseeded.
