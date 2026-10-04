# Experiment protocols

A **protocol** defines what the coding agent is asked to produce, which framework commands it can use, how interaction is controlled, and when the task is complete. Select one protocol per task with `protocol=...`.

This page describes the current architecture, including **CWM v5**. “v5” names the design generation, not a CLI value or an accepted CWM's `cwm_version`.

## Choose a protocol

There are three runnable protocols:

- **`policy_search`** (default): write and evaluate `solution.py`.
- **`vanilla`**: inspect recorded experience and run observation-based controllers through `RunController`.
- **`cwm`**: use the same managed environment and controller execution, with mandatory CWM validation and per-action prediction checks. Controllers receive parsed CWM States instead of observations.

| | `vanilla` | `cwm` | `policy_search` |
|---|---|---|---|
| Main artifacts | `controller.py` | `world_model/*.py`, `controller.py` | `solution.py` |
| Main commands | `RunController` | `UpdateCodeWorldModel`, `RunController` | `SubmitSolution`, optional `ExitTask` |
| Direct agent environment access | No | No | Through the environment client |
| `single_instance` | Supported | Supported | Rejected with the current on-environment evaluator |
| `multi_instance` | Fresh start per call | Fresh start per call | Fresh evaluation episodes |
| Independent final policy evaluation | No | No | Yes |

For the agent workflow, API and complete parameter tables, read [Managed execution](managed_protocols.md), then [CWM v5](cwm.md) for the additional modeling contract. Both problem config groups still default to `multi_instance`; explicitly select `problem.lifecycle=single_instance` for persistent interaction.

`controller.*` configures policy-search evaluation, not vanilla/CWM. `features.*` is also specific to policy search; managed protocols require `features=none`.

Protocol configuration lives in [conf/protocol/](../src/regact/conf/protocol/). CWM's planner is disabled by default and its instructions/files are omitted when disabled. Both managed protocols require OS-isolated submitted-code workers, even when the coding agent's own sandbox is disabled.

“Policy search” means developing and evaluating a reusable policy in `solution.py`. An experiment protocol is distinct from `agent.args.tool_protocol`, which selects the syntax used to invoke tools. `problem.lifecycle` controls environment persistence, not the lifetime of a controller object.

## Agent instructions and workspace helpers

All protocols use `PromptBuilder` for terminal syntax, command presentation and
optional verbalization hints. Each supplies its own workflow content. CWM wording
lives in `src/regact/protocols/cwm/prompts/`; the generated workspace contains:

- `docs/CWM_modeling_phase.md`: CWM implementation, validation, code isolation and shared limits.
- `docs/active_exploration_phase.md`: real controller submission, feedback and stopping conditions.
- `docs/plan_in_CWM.md`: optional goal-based planning.

The docstrings in `framework/data_api.py` document data access, returned fields and examples.
There is no additional CWM_INTERFACE.md. `world_model/` contains only agent-managed CWM files.

The system prompt gives the workflow and an inventory of the files actually supplied
for that game/configuration. The modeling and optional planner documents show relevant configured limits; runtime budget errors name their effective parameter and value. The game
section comes from the problem's prompt method; under CWM it uses observation
dictionaries and does not suggest policy-search submission commands.

Game helpers are supplied as `framework/arc_agi_helper.py` or
`framework/minigrid_helper.py`. Import them from `framework`, in any protocol.
CWM no longer creates an unused `code_library/`; policy search retains that directory
for its controller templates and agent-authored scripts. Existing run workspaces
and frozen submissions are not rewritten.

## Ownership

`protocols/managed/` owns shared environment lifetime, resets, initial collection, dataset access, isolated fresh controllers, action/time limits, metrics and tool transport. CWM specializes validation, planning and prediction hooks; vanilla uses observations directly. Vanilla does not inherit the CWM protocol. The existing policy-search evaluator remains separate.

The scientific comparison is **requiring and validating an explicit CWM versus not requiring one**. Vanilla agents remain free to write predictive code. Match agent/model, game, seed, lifecycle, collection settings and budgets when comparing protocols; CWM validation/prediction adds computation even with equal real-action budgets.

## Execution references

- [Managed execution](managed_protocols.md) defines call versus episode, single-/multi-instance behavior, explicit resets, controller callbacks, dataset access and all shared budget scopes.
- [CWM v5](cwm.md) defines the two phases, accepted snapshots, per-action validation, optional local simulation/planning, CWM-specific parameters and limitations.
- [Experiments](experiments.md) describes saved evidence and conversation/CWM playback. Call trajectories are reconstructed on demand from recorded observations.

## Runtime extension points

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
step before the agent starts. The default preparation is a no-op; both managed protocols use it for
random prefill. Dry runs skip preparation and agent execution. Protocols also choose whether the workspace exposes an environment
client (`exposes_environment`). The shared loop does not
interpret submission counts, scores, or CWM phases.

The order at a turn boundary remains: interruption, protocol outcome, turn limit,
tool-call limit, walltime limit. Mid-turn cooperative stops wait for all observed
tool calls to return. An idle remote agent stops promptly on an interrupt; a second
interrupt explicitly forces cancellation of outstanding work. The exit reason is saved before teardown hooks run; hook
failures are logged as before. Backend retries and the no-tool-progress breaker
retain their existing behavior.

## Preserving existing experiments

Existing policy-search launch commands retain their workflow. The explicit selection is:

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
`protocol=cwm features=none` and `protocol.*` parameters. The old feature implementation and verifier have been removed. Only the config
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

## Regression coverage

Vanilla and CWM assemble their briefs through [managed/prompting.py](../src/regact/protocols/managed/prompting.py) and its Markdown templates. Role, game instructions, terminal syntax, dataset access, preview images and execution rules are shared. [test_managed_prompt.py](../tests/test_managed_prompt.py) checks that only Working directory, Workflow and Framework commands differ for matching settings.

[test_protocols.py](../tests/test_protocols.py) covers the protocol boundary and retained policy-search behavior, including prompt compatibility. Evaluation tests cover failed episodes, perfect-score stopping, ExitTask, finalization, retries and outstanding tool completion. A toy protocol exercises the shared runner without policy artifacts.

[test_managed_protocol.py](../tests/test_managed_protocol.py) and [test_cwm_protocol.py](../tests/test_cwm_protocol.py) exercise lifecycle, resets, fresh controllers, validation, planning, counterexamples, deduplication, deadlines, isolated snapshots and replay. The `agent=alan_remote` preset enables manual interface checks without model API calls; see [Agents](agents.md).

## Feedback and diagnostics

Managed commands take no arguments and read fixed workspace files. Transport retry IDs and source hashes are retained for bookkeeping rather than presented as ordinary agent tasks. CWM validation returns Accepted / Refused / Incomplete; `cwm_version` and `dataset_version` identify accepted code and evidence, not quality scores.

A command that changes the phase reports it in its own result, as a `phase_change` field (`from`, `to`, `next_step`), so every agent reads one JSON document. A replay repeats no work and returns the stored result, phase change included. Keep-alive reminders use the same phase description as transitions.

First-time real milestones and evidence IDs appear in controller feedback. See [the data API](managed_protocols.md#dataset-api) for counters, pagination and images, and [CWM validation](cwm.md#interpret-validation-feedback) for counterexamples and error meanings.

## Flagging

Flagging warnings quote the originating command (up to 800 characters, with middle
truncation). Exact tool-call/result flags keep their call ID internally. A trusted
HTTP flag during concurrent calls cannot always be attributed to one command;
the warning lists candidate commands and states that uncertainty. Alan receives
warnings after completed tools; CLI agents receive queued warnings on their next
outer send cycle, if the task continues. `flagging_warning_cap=0` disables warning
messages but retains flag recording.

## Migration from earlier CWM versions

| Earlier interface | Current interface |
|---|---|
| Agent-operated initial collection / phase 0 | Framework prepares bounded random data before the agent starts |
| Numbered phases | **CWM Modeling** and **Active Exploration** |
| `SubmitExplorationController` | `RunController` |
| `exploration.py` | `controller.py` |
| Managed `framework/control.py` | `framework/commands.py` |
| `framework/cwm_data.py` | `framework/data_api.py` |
| `framework/simulation.py` or an agent-owned model environment | `framework/cwm_env.py` plus editable `simulate.py` |
| Preliminary simulated episode / novelty gate | Direct real execution with per-action checks; local simulation is optional |
| `protocol.execution.max_seconds_per_episode` | Removed; use `protocol.execution.max_seconds_per_RunController` |

Policy search retains its own `solution.py`, `framework/control.py` and evaluation settings. Existing saved runs/workspaces are not rewritten. Historical field names in their logs describe the implementation that produced them. Current limitations and deferred changes are listed in [CWM v5](cwm.md#scientific-and-operational-limitations).
