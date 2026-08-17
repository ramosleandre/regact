# Features

The **controller** is always-on core: every run has the agent write a pure
`act(obs) -> action` policy in `solution.py` and submit it (`SubmitSolution` / `ExitTask`),
scored by rolling episodes on the env. It is **not** a feature - see
[Controller](#controller) below.

A **feature** is an OPTIONAL capability layered on top of the controller. It bundles four
things: workdir **templates** (scaffolding), a **prompt fragment**, **tools** the agent can
call, and teardown **hooks**. regact ships one:

| `features=` | What it does | Scores on env? |
|---|---|---|
| `none` | no extra feature (the default) | - |
| `cwm` | Code World Model: records env transitions + an agent-run `verify.py` coherence check | no |

## Controller

The controller is configured under `controller.*` (group
[`conf/controller/`](../src/regact/conf/controller/)), not as a feature. Its knobs:
`n_episodes`, `max_moves`, `record_video`, `shadow_replay`.

```bash
# the always-on controller with 3 eval episodes, no video
make run ARGS="controller.n_episodes=3 controller.record_video=false"
```

## Use a feature

Each feature **owns its own knobs**, so run-level config stays small. Select a feature by
name and override a knob with its dotted path:

```bash
# add the Code World Model feature on top of the always-on controller
make run ARGS="features=cwm features.cwm.max_tested_transitions_per_verify=500"
```

The config groups in [`conf/features/`](../src/regact/conf/features/):

- **`none.yaml`** - no feature (the default).
- **`cwm.yaml`** — `max_tested_transitions_per_verify`,
  `max_printed_incoherence_transitions_per_verify`.

A feature is additive: `features=cwm` keeps the always-on controller (SubmitSolution/ExitTask)
and adds cwm on top. The `features:` mapping order is also the order env wrappers are applied
(first = innermost).

## Add a feature

A feature implements the [`Feature`](../src/regact/features/base.py) ABC;
[`cwm.py`](../src/regact/features/cwm.py) is the worked example (templates + a prompt
fragment + an env wrapper). The always-on controller lives in
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

Optional: `env_wrapper(ctx)` returns an `env -> wrapped env` factory applied server-side
(cwm uses it to record every transition). It must preserve the `WrappedEnv` surface.

**2. Register it** at the bottom of the module — features are string-keyed, no enum:

```python
register_feature(MyFeature.name, MyFeature)
```

Add it to `_load_builtins()` in [`features/base.py`](../src/regact/features/base.py).
`build_features` instantiates `MyFeature(**params)` from the config, so your knobs arrive
as constructor kwargs.

**3. Add a config group** `conf/features/<name>.yaml` writing its `features.<name>:` entry
with the knob defaults.
