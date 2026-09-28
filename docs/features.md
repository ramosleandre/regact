# Features

The **controller** is always-on core: every `policy_search` run has the agent write an
`act(obs) -> action` policy in `solution.py` and submit it (`SubmitSolution` / `ExitTask`),
scored by rolling episodes on the env. It is **not** a feature - see
[Controller](#controller) below. The controller may keep internal state between
actions; evaluation constructs a new controller for each episode.

A **feature** is an optional additive capability inside `policy_search`. It bundles
workspace templates, a prompt fragment, tools and teardown hooks. The default is
`features=none`; no additional built-in feature currently ships.

CWM is a separate [experiment protocol](protocols.md), selected with
`protocol=cwm features=none`. The retired `features=cwm` setting produces a migration
error. Its verifier and controller-coupled implementation have been removed.

## Controller

The controller is configured under `controller.*` (group
[`conf/controller/`](../src/regact/conf/controller/)), not as a feature. Its knobs:
`n_episodes`, `max_moves`, `n_videos`, `shadow_replay`, `exit_task_enabled`.

`n_videos` caps the number of final-evaluation episodes recorded (at most `n_episodes`);
set it to `0` to disable video. Numbered submissions are scored without recording video.

```bash
# the always-on controller with 3 eval episodes, no video
make run ARGS="controller.n_episodes=3 controller.n_videos=0"
```

## Add a feature

A feature implements the [`Feature`](../src/regact/features/base.py) ABC. The policy-search controller lives in
[`controller.py`](../src/regact/features/controller.py) - it uses the same
`templates`/`prompt_fragment`/`tools`/`hooks` seams but is core, built from `config.controller`,
not registered as a feature.

**1. Subclass `Feature`**, set `name`, and take your knobs as **constructor kwargs**:

```python
class MyFeature(Feature):
    name = "myfeature"
    evaluates_on_env = False        # True if you score by rolling episodes on the env

    def __init__(self, *, my_knob: int = 10) -> None:
        self._my_knob = my_knob
```

Implement the four abstract methods:

- `templates(ctx)` — files scaffolded into the workdir (a `TemplateFile` list).
- `prompt_fragment(ctx)` — markdown appended to the agent's brief (or `None`).
- `tools(deps)` — the [`Tool`](../src/regact/tools/base.py) objects the agent can call
  (each has `name`, `description`, `input_schema`, and `async call(args, ctx)`).
- `hooks(deps)` — [`Hook`](../src/regact/features/base.py) objects fired at their phase
  (currently `TEARDOWN` — e.g. re-scoring the final submission).

Both `tools` and `hooks` receive a [`RunDeps`](../src/regact/features/base.py): the agnostic
`env_client`, the solution/submissions paths, the metric callables, the seed, etc. — the
runtime dependencies the orchestrator owns.

Optional extension points:

- `env_wrapper(ctx)` returns an `env -> wrapped env` factory applied server-side
  and must preserve the `WrappedEnv` surface.
- `submission_metrics(deps)` returns a JSON-serializable metric mapping after an
  evaluation. It is stored under `results.json` → `features` → the feature's name;
  this is how a feature contributes metrics without replacing the controller score.

**2. Register it** at the bottom of the module — features are string-keyed, no enum:

```python
register_feature(MyFeature.name, MyFeature)
```

Add it to `_load_builtins()` in [`features/base.py`](../src/regact/features/base.py).
`build_features` instantiates `MyFeature(**params)` from the config, so your knobs arrive
as constructor kwargs.

**3. Add a config group** `conf/features/<name>.yaml` writing its `features.<name>:` entry
with the knob defaults.
