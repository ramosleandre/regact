# Environments

An **environment** (a "problem") is the game the agent plays. A problem exposes one or
more **tasks** (games/levels) and knows how to build the env, render it, prompt about it,
and score an episode. regact ships two:

| `problem=` | Tasks | Lifecycle | Needs |
|---|---|---|---|
| `arc_agi` | the discovered ARC-AGI-3 games | `single_instance` | `make install-arc` |
| `minigrid` | MiniGrid gym environments | `multi_instance` | `make install-minigrid` |

## Use an environment

Pick a problem by group name; `tasks` selects which games (empty = all).

```bash
# ARC-AGI-3, one game
make run ARGS="agent=alan problem=arc_agi 'problem.tasks=[ls20]'"

# MiniGrid, a specific env, fully observable
make run ARGS="problem=minigrid 'problem.tasks=[MiniGrid-DoorKey-5x5-v0]' problem.kwargs.fully_obs=true"
```

The `ProblemConfig` fields are `name`, `tasks`, `lifecycle`, `obs_mode`, `info_mode`,
`seed`, and `kwargs` (environment-construction options). The config groups in
[`conf/problem/`](../src/regact/conf/problem/):

| File | `tasks` | `kwargs` |
|---|---|---|
| `arc_agi.yaml` | `[]` (all) | `operation_mode: offline`, `environments_dir` |
| `minigrid.yaml` | `[MiniGrid-Empty-5x5-v0]` | `fully_obs: false` |
| `minigrid_lite.yaml` | the curated 20 | `fully_obs: true` |
| `minigrid_full.yaml` | all 72 | `fully_obs: true` |

**Lifecycle** — `multi_instance` builds a **fresh env per episode** (more resets = better
stats); `single_instance` keeps **one env per game** (RESET = level reset; this is ARC).
A single-instance problem paired with a feature that scores on the env is **refused** at
startup — exploration and evaluation would share the same env, making the score
session-level rather than an isolated policy. See [`arc_agi.yaml`](../src/regact/conf/problem/arc_agi.yaml).

## Add an environment

A problem implements the [`BaseProblem`](../src/regact/problems/base.py) ABC.
[`MiniGridProblem`](../src/regact/problems/minigrid/problem.py) is a compact example.

**1. Subclass `BaseProblem`**, set `name`, and implement the abstract methods:

- `make_env(task_name)` — return a gym-like env (`reset()`, `step(action)`; import the
  game library **lazily** here so the module loads without the extra installed).
- `get_task_names()` — the tasks this problem exposes.
- `obs_renderer(task_name, *, mode)` — an `ObsRenderer` turning an obs into what the agent
  sees.
- `compute_episode_metrics(final_obs, *, steps)` and `aggregate_episode_metrics(episodes)`
  — the per-episode score and its aggregate.
- `build_prompt(task_name, *, info_mode)` — the game briefing (keep the prose in a markdown
  file next to the module).
- `config_kwargs()` — kwargs to rebuild the problem for trusted-side eval.

Optional hooks (each has a default): `milestone_detector`, `helper_templates`,
`secret_modules` (the packages that ARE the game — hidden from the sandbox),
`render_frame` (obs → RGB frame for the video), `render_obs_text`,
`derived_submission_metrics` (offline scores like ARC's RHAE, shown in the viewer).

**2. Register it** at the bottom of the module — problems are string-keyed, no enum:

```python
register_problem("mygame", lambda kwargs: MyGameProblem(**kwargs))
```

Add it to `_load_builtins()` in [`problems/base.py`](../src/regact/problems/base.py) so it
self-registers, and the factory splats `config.problem.kwargs` into your constructor.

**3. Add a config group** `conf/problem/<name>.yaml` with `name`, `tasks`, `lifecycle`,
and any `kwargs`.

> **Env wrappers.** Features can wrap the env server-side (see
> [Features](features.md) — `env_wrapper`), applied in `features:` list order. A wrapper
> must preserve the [`WrappedEnv`](../src/regact/env/wrapper.py) surface
> (`reset`/`step`/`close`, `action_count`, `last_obs`).
