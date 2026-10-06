# Experiments

How to launch runs, where the artifacts land, and how to inspect them.

## Launch a run

A run is composed by Hydra: pick `agent`, `problem` and `protocol`, then override fields on the CLI. `features` and `controller` configure additions/evaluation inside policy search. Vanilla/CWM use `protocol.*` and require `features=none`.

```bash
# smoke test: scripted agent, no LLM; requires ARC ls20 and its engine/data
make run ARGS="experiment=dev"

# persistent vanilla interaction
make run ARGS="agent=claude problem=arc_agi 'problem.tasks=[ls20]' protocol=vanilla features=none problem.lifecycle=single_instance limits.max_tool_calls=100 limits.max_seconds_per_task=3600"

# the same execution model, with CWM v6 requirements
make run ARGS="agent=codex problem=arc_agi 'problem.tasks=[ls20]' protocol=cwm features=none problem.lifecycle=single_instance limits.max_tool_calls=100 limits.max_seconds_per_task=3600"

# see the composed config without running it
make run ARGS="agent=claude problem=arc_agi protocol=vanilla --cfg job --resolve"
```

The repository defaults to `protocol=policy_search` and `problem.lifecycle=multi_instance`. Select persistent interaction explicitly. [Managed execution](managed_protocols.md#shared-protocol-parameters) lists every shared protocol parameter; [CWM](cwm.md#additional-cwm-parameters) lists its extra settings. Launcher overrides can differ from these defaults; the saved `config.json` records the actual run configuration.

### Preview without model calls

`--cfg job --resolve` prints configuration only. `dry_run=true` instead creates a task workspace and records the generated prompt/first message for `make viz`, without launching the agent or running initial random collection:

```bash
make run ARGS="agent=codex problem=arc_agi 'problem.tasks=[ls20]' protocol=cwm features=none problem.lifecycle=single_instance experiment_name=cwm-preview dry_run=true"
make viz EXP=experiments/cwm-preview/latest
```

A dry run still constructs framework/environment resources and needs the selected engine and supported isolation setup. It verifies prompt/workspace generation, not actual controller execution or backend authentication. Set `dry_run=false` for a real benchmark.

### Profiles

A profile bundles a whole setup (agent + problem + limits) under one name. They live in
[`conf/experiment/`](../src/regact/conf/experiment/):

- **`dev`** — plumbing smoke test with the scripted agent on ARC `ls20`; no LLM, but
  requires `make install-arc` and the local game data.
- **`research`** — a real ARC run with a coding CLI, anti-cheat on, video recorded.
- **`competition`** — legacy Kaggle profile; its policy-search/single-instance combination is rejected (see below).

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
      experiment_state.json           # live state (saved atomically per event); agent_usage =
                                      # Claude/Codex token totals per model, set at close
      events.jsonl / output.log       # the operational log
    workdir/                          # the agent's working directory
```

Policy search adds:

```text
workdir/submissions/
  0/results.json                      # first numbered submission (then 1/, 2/, ...)
  final/
    results.json                      # final evaluation of the current controller
    video_0.mp4                       # optional (then video_1.mp4, ...)
```

Vanilla and CWM instead add these paths inside the task/attempt directory:

```text
cwm/                                  # shared storage name, also used by vanilla
  status.json                         # latest protocol state and summaries
  experience.sqlite3                  # dataset, ordered occurrences and operation evidence
  bundles/                            # immutable submitted Python code and manifests
workdir/
  controller.py
  framework/
  tmp/images/                         # bounded, replaceable observation previews
  world_model/                        # CWM only
  plans/                              # saved planner outputs when used
```

No policy-search `submissions/final` re-score runs for vanilla/CWM. Their metrics describe real interaction, with latest/best results retained alongside complete operation records. SQLite deduplicates observations/transitions while retaining repeated occurrences for trajectory reconstruction. Reset boundaries and controller-call boundaries are separate.

With `n_attempts_per_task > 1`, the same `config.json`, `logs/`, and `workdir/` layout
lives under `<task_name>/attempt_0/`, `attempt_1/`, etc. Attempts are scheduled across
tasks in rounds.

For policy search, `final` is a new evaluation of the controller left in the workdir at teardown, not a
copy of the best numbered submission. Videos are recorded there when `n_videos > 0`
and frames can be rendered/encoded. There is no `submissions/last` directory in the
current writer.

The state file is saved atomically as events are processed. Normal timestamped runs
claim separate directories; callers supplying an explicit run directory are responsible
for avoiding reuse.

## The visualizer

Browse runs, transcripts and protocol-specific evidence:

```bash
make viz EXP=experiments/<experiment_name>/latest        # PORT=8030 by default
```

The viewer reads logs, workspace artifacts, policy-search results and managed protocol records directly. It does not control the agent or resume a finished task. Refresh a running view to inspect newly written evidence.

| Panel / control | What to inspect |
|---|---|
| **Overview** | Configuration, status, resource use, game preview and metrics |
| **Conversation** | Agent messages and tool calls; framework feedback and warning flags |
| **Jump to** | Individual framework command calls, including multiple calls in one Codex turn |
| **Load controller playback** | The real observation sequence for that vanilla/CWM call, loaded next to its result |
| **CWM** | CWM phase history, dataset, validation, counterexamples, real episodes and optional plans |
| **Artifacts / Logs** | Workspace files, saved outputs and operational evidence |
| **Graphs** | Available task/benchmark metrics and aggregations |

Command colors distinguish `UpdateCodeWorldModel` (green), `PlanInCWM` (purple), and `RunController` (blue). A command's success marker does not mean the full game was solved. Warning flags remain independent of command badges.

Real playback follows ordered observation occurrences, preserving repeated IDs. It reconstructs frames on demand and stores no video. CWM counterexamples provide visual comparisons plus raw/structural evidence; metadata differences may be invisible in images. Plan playback runs the saved action list in its frozen CWM; it does not rerun search or touch the real environment. Source/runtime mismatches can prevent predicted replay.

Vanilla has call playback without the CWM-specific panel. Historical runs without call-level sequences cannot supply the new call player. Images in ordinary tool results appear only when the backend delivered image content in its normalized events: a printed path is not an image, and some CLI image events are absent from the transcript. Analyst playback does not imply the agent saw those images.

The dataset stores complete observations, but the API and feedback expose bounded pages/previews. Neither the preview selection nor the current workspace alone is the full historical evidence. In particular, inspect the frozen source associated with a command when the workspace has since changed.

### Interpreting benchmark results

Match agent/model, lifecycle, seeds, initial collection and budgets when comparing vanilla with CWM. Report tool calls, real actions, time and game progress separately. Equal tool-call counts need not mean equal real actions or compute, and one controller call need not equal one episode.

Managed protocols measure online interaction; their best observed result is not an independent evaluation on held-out seeds. Policy-search submissions and final evaluation also reuse configured seed sequences unless you explicitly set up a separate evaluation. Graphs summarize recorded metrics; they do not establish CWM correctness, useful novelty, or generalization. See [CWM limitations](cwm.md#scientific-and-operational-limitations), including hidden-state effects in MiniGrid.

## Competition (Kaggle)

**Legacy integration:** the default `competition` profile combines `single_instance` with policy search, whose on-environment evaluator rejects that lifecycle. Vanilla/CWM now support persistent environments, but that does not establish compatibility with the Kaggle adapter or online ARC reset scopes. The entry point below is not a validated v5 competition launch.

The Kaggle path uses argparse instead of Hydra:

```bash
make run-kaggle ARGS="--games ls20 ft09 --parallel 2"
```

Flags: `--config` (profile), `--games` (override tasks), `--parallel`, `--output-root`,
`--agent` (swap the backend name, keeping the profile's model/base_url/args). On an ARC run
it prints the RHAE summary at the end. See the
[Kaggle integration](../src/regact/kaggle/) for the notebook and serving code.

## HPC

Ready-to-submit isolation diagnostics for the two validated clusters:

- [`scripts/adastra/`](../scripts/adastra/) — probe + a SimpleLM-served ARC run.
- [`scripts/jeanzay/`](../scripts/jeanzay/) — the isolation probe.

Both confine the agent with bwrap (`sandbox=true`); see [Sandboxing](sandboxing.md).
