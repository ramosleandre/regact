# Experiments

How to launch runs, where the artifacts land, and how to inspect them.

## Launch a run

A research run is composed by Hydra: pick the `agent` / `problem` / `features` groups and
override any field on the CLI.

```bash
# smoke test: scripted agent, no LLM; requires ARC ls20 and its engine/data
make run ARGS="experiment=dev"

# a real run
make run ARGS="agent=alan problem=arc_agi 'problem.tasks=[ls20]' controller.n_episodes=3"

# see the composed config without running it
make run ARGS="agent=claude problem=arc_agi --cfg job"
```

### Profiles

A profile bundles a whole setup (agent + problem + limits) under one name. They live in
[`conf/experiment/`](../src/regact/conf/experiment/):

- **`dev`** — plumbing smoke test with the scripted agent on ARC `ls20`; no LLM, but
  requires `make install-arc` and the local game data.
- **`research`** — a real ARC run with a coding CLI, anti-cheat on, video recorded.
- **`competition`** — legacy Kaggle profile; currently rejected by the runner (see below).

The `dev` and `research` profiles **select** groups, so a CLI override still wins:
`make run ARGS="experiment=research agent=codex"` runs codex, not the profile's default.
The legacy `competition` profile uses inline agent/problem fields instead; combining it
with an `agent=` override can produce a hybrid configuration.

## Outputs

By default, each run gets a fresh **timestamped** directory.
`<output_root>/<experiment_name>/latest` is a symlink to that directory when the
filesystem supports it. It identifies a run, not a submission.

With `n_attempts_per_task=1`, each task has this layout:

```
<output_root>/<experiment_name>/<timestamp>/
  <task_name>/
    config.json                       # the run config (api_key redacted)
    logs/
      transcript.jsonl                # the normalized agent event stream
      experiment_state.json           # live state (saved atomically per event)
      events.jsonl / output.log       # the operational log
    workdir/                          # the agent's working directory
      submissions/
        0/results.json                # first numbered submission (then 1/, 2/, ...)
        final/
          results.json                # final evaluation of the current controller
          video_0.mp4                 # optional final-evaluation video (then video_1.mp4, ...)
```

With `n_attempts_per_task > 1`, the same `config.json`, `logs/`, and `workdir/` layout
lives under `<task_name>/attempt_0/`, `attempt_1/`, etc. Attempts are scheduled across
tasks in rounds.

`final` is a new evaluation of the controller left in the workdir at teardown, not a
copy of the best numbered submission. Videos are recorded there when `n_videos > 0`
and frames can be rendered/encoded. There is no `submissions/last` directory in the
current writer.

The state file is saved atomically as events are processed. Normal timestamped runs
claim separate directories; callers supplying an explicit run directory are responsible
for avoiding reuse.

## The visualizer

Browse a run in the browser — transcript grouped by turn, submissions, scores, videos:

```bash
make viz EXP=experiments/<experiment_name>/latest        # PORT=8030 by default
```

It reads the run dir directly (`logs/` + `workdir/submissions/`) and, for problems that
define them, shows derived offline metrics (e.g. ARC's RHAE).

## Competition (Kaggle)

**Currently incompatible with the runner:** the default `competition` profile selects
`single_instance`, which the always-on controller rejects. The command below documents
the existing entry point, but does not currently launch a supported run with that profile.
Changing it to `multi_instance` would change the evaluation semantics; it would not
restore persistent-session competition support.

The Kaggle path uses argparse instead of Hydra:

```bash
make run-kaggle ARGS="--games ls20 ft09 --parallel 2"
```

Flags: `--config` (profile), `--games` (override tasks), `--parallel`, `--output-root`,
`--agent` (swap the backend name, keeping the profile's model/base_url/args). On an ARC run
it prints the RHAE summary at the end. See the
[arc-agi-3 skill](../src/regact/kaggle/) for the notebook and serving details.

## HPC

Ready-to-submit isolation diagnostics for the two validated clusters:

- [`scripts/adastra/`](../scripts/adastra/) — probe + a SimpleLM-served ARC run.
- [`scripts/jeanzay/`](../scripts/jeanzay/) — the isolation probe.

Both confine the agent with bwrap (`sandbox=true`); see [Sandboxing](sandboxing.md).
