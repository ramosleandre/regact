# Environments

An **environment** (a "problem") is the game the agent plays. A problem exposes one or
more **tasks** (games/levels) and knows how to build the env, render it, prompt about it,
and score an episode. regact ships two:

| `problem=` | Tasks | Default lifecycle | Needs |
|---|---|---|---|
| `arc_agi` | the discovered ARC-AGI-3 games | `multi_instance` | `make install-arc` |
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
`helper`, `seed`, and `kwargs` (environment-construction options). The config groups in
[`conf/problem/`](../src/regact/conf/problem/):

| File | `tasks` | `kwargs` |
|---|---|---|
| `arc_agi.yaml` | `[]` (all) | `operation_mode: offline`, `environments_dir` |
| `minigrid.yaml` | `[MiniGrid-Empty-5x5-v0]` | `fully_obs: true` |
| `minigrid_lite.yaml` | the curated 20 | `fully_obs: true` |
| `minigrid_full.yaml` | the full configured task list | `fully_obs: true` |

## Lifecycle and protocol

Both problem groups default to `multi_instance`. Support depends on the selected [protocol](protocols.md):

| Lifecycle | Vanilla / CWM | Policy search |
|---|---|---|
| `single_instance` | Preserve the live environment across initial collection, controller calls and CWM repair | Rejected: exploration and on-environment evaluation would share a session |
| `multi_instance` | Reset before each `RunController`; CWM requires the original initial observation | Evaluate policies on fresh episodes |

Every managed call creates a new controller object in either lifecycle. `single_instance` preserves the **environment**, not the Python controller's private memory. Set `problem.lifecycle=single_instance` explicitly to use it.

In vanilla/CWM `single_instance`, `ResetEnvironment` restarts the whole environment. Offline ARC also supports `ResetLevel`, preserving completed levels. `multi_instance` has no reset commands, since every `RunController` starts fresh. An observation with `is_done=True` ends a controller call; full-game success ends the task, while an unsuccessful terminal episode permits inspection/reset. See [Managed execution](managed_protocols.md#explicit-resets) for action accounting and dataset boundaries.

For policy search, more episodes do not automatically mean more varied evaluation: MiniGrid uses an
episode seed sequence, while deterministic ARC games ignore the seed. Submissions and
final evaluation reuse the configured seed sequence; these are not automatically
held-out tests. Managed protocols have no independent final evaluation. A fixed `problem.seed` also matters for reproducible multi-instance CWM resets.

## Observations and CWM compatibility

The normalized observation contains `frame`, `reward`, `is_done`, `available_actions` and `info`. `frame` contains the last frame rather than an animation history. In managed code these are dictionary fields. Milestones describe events from the preceding action, not cumulative progress.

CWM validates the complete observation, not just its image. Its current assumption is that observation plus action determines the next observation. A fully visible grid does not necessarily satisfy that assumption: MiniGrid's hidden step counter affects terminal reward and truncation, even with `fully_obs=true`. CWM stops if identical complete observation/action inputs have conflicting recorded successors. Vanilla makes no such prediction requirement. See [CWM limitations](cwm.md#scientific-and-operational-limitations) before interpreting failures as modeling mistakes.

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
- `build_prompt(task_name, *, info_mode, obs_mode, direct_interaction)` — the game briefing (keep the prose in a markdown
  file next to the module).
- `config_kwargs()` — kwargs to rebuild the problem for trusted-side eval.

Optional hooks (each has a default): `milestone_detector`, `helper_templates`,
`secret_modules` (the packages that ARE the game — hidden from the sandbox),
`render_frame` (obs → RGB frame for the video), `render_obs_text`,
`derived_submission_metrics` and `derived_trace_metrics` (scores a game derives from its own
data, like ARC's RHAE and LRHAE-Uncapped), `main_metrics` (which score keys are the game's main
ones: each run records them in `logs/experiment_state.json` and its end-of-run log line, and the
viewer shows them first; ARC uses `rhae`, `lrhae_uncapped` and `relative_env_actions`),
`failure_metrics(*, steps)` (zero credit for controller errors), and
`is_perfect(aggregate)` (whether a submission should end the run early). The default
perfect predicate checks `success_rate >= 1.0`; override it if your problem uses a
different completion metric (ARC uses `win_rate`). In policy search, the loop also requires a complete,
error-free evaluation before stopping as solved.

For managed protocols, also review these extension points:

- `reset_commands()` advertises explicit reset capabilities; the default is `ResetEnvironment`. Additional commands need corresponding trusted reset handling.
- `validate_controller_action(action)` rejects control operations that must go through an explicit command, such as ARC reset actions.
- `enumerate_actions(obs)` supplies a finite public action space for initial random collection and optional planning. Parameterized actions need a problem-specific implementation; the default enumerates integer action IDs.
- `milestone_kind(name)` classifies milestones as progress, failure or other events; `exploration_score(aggregate)` ranks the best observed managed result.

`helper_templates(..., direct_interaction=False)` and the same prompt flag let a problem describe observations/actions without suggesting direct environment access in managed protocols. Helpers must not reveal game implementation code.

Override `failure_metrics` when your problem has additional score fields. In policy search, failed
controller episodes remain in the scoring denominator: MiniGrid assigns no success
or reward; ARC assigns no success or level completion, including any partial progress
before the error. Step counts remain diagnostic. Environment/harness failures instead
mark the evaluation incomplete (`evaluation_complete=false`); `n_expected_episodes`
records the requested count, and `n_errors` counts all failed episodes.

Validate actions before calling the game engine and raise
`regact.envclient.errors.InvalidActionError` for invalid input. The server/client
preserves this distinction, so a malformed action is a controller failure even when
validation happens inside the environment. Other engine/transport errors are not
assumed to be controller faults.

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
> must preserve the [`WrappedEnv`](../src/regact/env/wrapped_env.py) surface
> (`reset`/`step`/`close`, `action_count`, `last_obs`).
