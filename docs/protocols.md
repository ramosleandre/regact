# Experiment protocols

## Status

There are two runnable protocols:

- **`policy_search`** (default): write and evaluate `solution.py`.
- **`cwm`**: pre-fill real observations with a bounded random policy, start the
  agent with model construction, then validate the model and simulation-check
  exploration controllers before running them in a fresh real environment.

Select CWM with `protocol=cwm features=none`. Its options are in
[`conf/protocol/cwm.yaml`](../src/regact/conf/protocol/cwm.yaml). It requires
`problem.lifecycle=multi_instance` and OS-isolated code workers. Its three tools
are UpdateCodeWorldModel, PlanInCWM and SubmitExplorationController. It does not
register SubmitSolution or ExitTask and does not perform a final policy re-score.

CWM initialization targets `protocol.n_unique_observations_in_initial_collection=20`, bounded by
`protocol.max_actions_per_initial_collection=1000` and
`protocol.max_seconds_per_initial_collection=30`. Action selection uses the problem
seed (0 if unspecified); environment resets still use `problem.seed` as configured.
All initial actions are logged and charged to the total real-action budget.
The agent starts in **CWM Modeling**, even if a collection cap stops prefill below target.
The other phase is **Active Exploration**; initialization is not an agent phase.
The current dataset API hides prefill targets and uses explicit unique/total counters.
CWM workspaces expose data/control/exploration helpers, without `make_env.py` or
`cwm_client.py`; direct environment endpoints reject and flag calls in every phase. Exploration feedback provides compact observation/transition ID ranges and up to `protocol.n_tmp_images_saved_per_exploration=8` temporary PNG previews.

`protocol.workspace_helpers_enabled=true` additionally supplies `framework/simulation.py`,
containing the local `EnvCWM` environment, its convenience factory and a controller runner.
These use current workspace code; acceptance and real interaction remain the
responsibility of the three framework tools.

“Policy search” means that the code agent develops and evaluates a policy in
`solution.py`. “Controller” remains the name of that Python artifact and its
existing `controller.*` evaluation settings. An experiment protocol is different
from `agent.args.tool_protocol`, which selects the syntax used to invoke tools.

## Agent instructions and workspace helpers

Both protocols use `PromptBuilder` for terminal syntax, command presentation and
optional verbalization hints. Each supplies its own workflow content. CWM wording
lives in `src/regact/protocols/cwm/prompts/`; the generated workspace contains:

- `docs/CWM_modeling_phase.md`: CWM implementation, validation, code isolation and shared limits.
- `docs/active_exploration_phase.md`: controller submission and two-stage feedback.
- `docs/plan_in_CWM.md`: optional goal-based planning.

The docstrings in `framework/data_api.py` document data access, returned fields and examples.
There is no additional CWM_INTERFACE.md. `world_model/` contains only agent-managed CWM files.

The system prompt gives the workflow and an inventory of the files actually supplied
for that game/configuration. Phase documents show the configured limits. The game
section comes from the problem's prompt method; under CWM it uses observation
dictionaries and does not suggest policy-search submission commands.

Game helpers are supplied as `framework/arc_agi_helper.py` or
`framework/minigrid_helper.py`. Import them from `framework`, in either protocol.
CWM no longer creates an unused `code_library/`; policy search retains that directory
for its controller templates and agent-authored scripts. Existing run workspaces
and frozen submissions are not rewritten.

## Ownership

| Shared runtime | Selected protocol |
| --- | --- |
| Launcher, scheduling, problem/agent construction | Workflow validation and workspace artifacts |
| Agent sandbox, HTTP transport and tool dispatch | Instructions and available framework tools |
| Transcripts, common state persistence, operational logs | Mutable workflow state, such as phases |
| Turn/tool/walltime limits and backend-error retries | Completion criteria and keep-alive reminders |
| Waiting for outstanding tools before a cooperative stop | Finalization hooks and workflow diagnostics |

The runner selects one `ExperimentProtocol` per task. After the environment and
workspace exist, `bind(ProtocolContext)` creates its `ProtocolSession`. A session
provides tools, hooks, a reminder, a stop reason, and an optional `prepare(stop)`
step before the agent starts. The default preparation is a no-op; CWM uses it for
random prefill. Protocols also choose whether the workspace exposes an environment
client (`exposes_environment`). The shared loop does not
interpret submission counts, scores, or CWM phases.

The order at a turn boundary remains: interruption, protocol outcome, turn limit,
tool-call limit, walltime limit. Mid-turn cooperative stops wait for all observed
tool calls to return. An idle remote agent stops promptly on an interrupt; a second
interrupt explicitly forces cancellation of outstanding work. The exit reason is saved before teardown hooks run; hook
failures are logged as before. Backend retries and the no-tool-progress breaker
retain their existing behavior.

## Preserving existing experiments

Existing launch commands require no changes. The explicit selection is:

```sh
python -m regact.run_exp protocol=policy_search agent=codex problem=minigrid
```

Policy-search workflow logic, SubmitSolution/ExitTask behavior, evaluation, result
paths and final re-score are preserved. Shared prompt assembly retains its existing
policy-search wording; game helper paths have moved to `framework/` as described above.
Run configuration artifacts additionally record `protocol: {name: policy_search}`.
The executor and submission implementation are reused unchanged. Alan's remote scripted mode remains available for inspecting the agent interface
without a model call.

The feature mechanism is composed **inside policy_search**. It remains an
extension point for additive capabilities, not alternative workflows. The old
`features=cwm` launch is retired and fails with a migration message: use
`protocol=cwm features=none` and `protocol.*` parameters. The old implementation, verifier and launchers have been removed. Only the config
sentinel remains so old invocations receive that migration message.

## Adding another protocol

1. Implement `ExperimentProtocol` in `src/regact/protocols/`. Keep each task's
   mutable state local to its protocol/session; factories must return fresh instances.
2. Validate its own options and supported lifecycle. Supply its templates, prompt,
   tools, completion logic, reminders and finalization. It need not expose
   `solution.py`, SubmitSolution, ExitTask, or perform controller evaluation.
3. Add its built-in factory in `protocols/registry.py` (or register a factory before
   launching), and a Hydra group under `conf/protocol/`. Extra YAML fields go under
   `protocol.*`; the loader passes them as `ProtocolConfig.options` for that protocol
   to validate. Unknown protocols and unsupported options must fail clearly.
4. Exercise it through `run_task`, including tool completion, common limits,
   teardown, and concurrent task isolation. A new protocol should not require
   special-case branches in the common agent loop.

Protocols can configure trusted environment handlers/data routes before
bootstrap, receive the common session start clock, and close their resources even
on startup failure. CWM uses these hooks for its phase gate, serialized real
operations, SQLite experience store and immutable worker bundles. Model and
controller/goal execution are isolated separately, with no experience DB, agent
workspace or network access. The common loop remains unaware of phase semantics.

CWM's viewer tab reads its database and reconstructs real episodes or saved
predicted paths on demand; it does not store videos or reinterpret explorations
as policy-search submissions.

## Regression evidence

`tests/test_protocols.py` checks 60 supported prompt variants and two workspace layouts
against SHA-256 fingerprints captured from pre-refactor commit
`728aa2079aeaa947cb8cb215a481160995d99d9a`. The matrix covers every tool syntax,
ExitTask on/off, both lifecycle prompt variants and the optional
verbalization hints. The fingerprint comparison allows the explicit numeric-reward wording update from September 28; all other shared policy-search prompt text is checked unchanged. Testing single-instance text does not enable its rejected
on-environment evaluation configuration.

The existing integration tests cover evaluation failures, perfect-score stopping,
ExitTask, finalization, retries and waiting for enclosing/parallel tool results.
An independent toy protocol test verifies that the common runner can operate
without controller artifacts or submission semantics and still enforces limits.

`tests/test_cwm_protocol.py` exercises phase boundaries, validation, planning,
fresh simulation/real controllers, counterexamples, request deduplication, deadlines,
shutdown, immutable snapshots, filesystem isolation and reconstructed playback.
`scripts/local/cwm_v4_remote.sh` provides a no-token-cost Alan remote walkthrough.

## Feedback and diagnostics

CWM tool arguments are empty: fixed world-model files, `goal.py`, and `exploration.py`.
Validation returns Accepted / Refused / Incomplete. Accepted models use stable
`cwm_version` values; internal dataset freshness uses `dataset_version`. Transport
retry IDs and source-bundle hashes are not ordinary agent feedback.

The coordinator produces phase notices centrally. Tool output carries optional
messages; CLI-backed agents see them after the JSON in their shell result, while
native dispatch injects them separately. A replay repeats no work or phase notice.
First-time real milestones appear in the exploration result with evidence IDs.

The CWM viewer compares diagnostic images and retains raw evidence. Tool-result
images are shown only when their bytes were delivered in the backend event; a
printed file path is not an image delivery. Some backends omit native image events.

All Regact-owned `max_*` settings accept `null` to disable that cap. `limits.max_turns_per_task` counts outer agent send cycles; `limits.max_actions_per_episode` renews on reset. CWM simulation and real episodes have independent `protocol.execution.max_seconds_per_episode` allowances. Data byte limits guard bulk queries; single records/images remain readable.

Flagging warnings quote the originating command (up to 800 characters, with middle
truncation). Exact tool-call/result flags keep their call ID internally. A trusted
HTTP flag during concurrent calls cannot always be attributed to one command;
the warning lists candidate commands and states that uncertainty. Alan receives
warnings after completed tools; CLI agents receive queued warnings on their next
outer send cycle, if the task continues. `flagging_warning_cap=0` disables warning
messages but retains flag recording.
