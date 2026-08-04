# Features

A **feature** is what the agent builds and how it's scored. A feature bundles four things:
workdir **templates** (scaffolding), a **prompt fragment**, **tools** the agent can call,
and teardown **hooks**. regact ships two:

| `features=` | What it does | Scores on env? |
|---|---|---|
| `controller` | the agent writes a pure `act(obs) -> action` policy and submits it | yes |
| `cwm` | Code World Model: records env transitions + an agent-run `verify.py` coherence check | no |

## Use a feature

Each feature **owns its own knobs**, so run-level config stays small. Override a knob with
its dotted path:

```bash
# controller with 3 eval episodes, no video
make run ARGS="features=controller features.controller.n_episodes=3 features.controller.record_video=false"

# the Code World Model feature
make run ARGS="features=cwm"
```

The config groups in [`conf/features/`](../src/regact/conf/features/):

- **`controller.yaml`** — `n_episodes`, `max_moves`, `record_video`, `shadow_replay`.
- **`cwm.yaml`** — `max_tested_transitions_per_verify`,
  `max_printed_incoherence_transitions_per_verify`.

### Composing features

The `features=` groups are independent — `features=cwm` gives you **only** cwm (no
controller, so no SubmitSolution/ExitTask; such a run ends on a limit). To stack them,
list both — on the CLI or in a profile:

```bash
make run ARGS="+features.controller={} +features.cwm={}"
```

The `features:` list order is also the order env wrappers are applied (first = innermost).

## Add a feature

A feature implements the [`Feature`](../src/regact/features/base.py) ABC.
[`controller.py`](../src/regact/features/controller.py) and
[`cwm.py`](../src/regact/features/cwm.py) are the two examples — the first adds tools +
a scoring hook, the second adds an env wrapper.

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
