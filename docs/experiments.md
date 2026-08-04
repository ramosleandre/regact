# Experiments

How to launch runs, where the artifacts land, and how to inspect them.

## Launch a run

A research run is composed by Hydra: pick the `agent` / `problem` / `features` groups and
override any field on the CLI.

```bash
# the CI smoke: scripted agent, no LLM, no game
make run ARGS="experiment=dev"

# a real run
make run ARGS="agent=alan problem=arc_agi 'problem.tasks=[ls20]' features.controller.n_episodes=3"

# see the composed config without running it
make run ARGS="agent=claude problem=arc_agi --cfg job"
```

### Profiles

A profile bundles a whole setup (agent + problem + limits) under one name. They live in
[`conf/experiment/`](../src/regact/conf/experiment/):

- **`dev`** — fastest end-to-end, no LLM/no game (plumbing smoke test).
- **`research`** — a real ARC run with a coding CLI, anti-cheat on, video recorded.
- **`competition`** — the Kaggle profile (see below).

A profile **selects** groups, so a CLI override still wins:
`make run ARGS="experiment=research agent=codex"` runs codex, not the profile's default.

## Outputs

Each run gets a fresh **timestamped** directory, with a `latest` symlink pointing at it:

```
<output_root>/<experiment_name>/<timestamp>/
  <task_name>/
    config.json                       # the run config (api_key redacted)
    logs/
      transcript.jsonl                # the normalized agent event stream
      experiment_state.json           # live state (saved atomically per event)
      events.jsonl / output.log       # the operational log
    workdir/                          # the agent's working directory
      submissions/<n|final>/results.json   # each scored submission (+ .mp4)
```

The state file is written **atomically after every event**, so it always reflects the last
turn — even after a crash. A re-run never overwrites an old one (new timestamp).

## The visualizer

Browse a run in the browser — transcript grouped by turn, submissions, scores, videos:

```bash
make viz EXP=experiments/<experiment_name>/latest        # PORT=8030 by default
```

It reads the run dir directly (`logs/` + `workdir/submissions/`) and, for problems that
define them, shows derived offline metrics (e.g. ARC's RHAE).

## Competition (Kaggle)

The Kaggle path uses argparse instead of Hydra, driven by the `competition` profile:

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
