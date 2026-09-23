# Experiment protocols

## Status

There are two runnable protocols:

- **`policy_search`** (default): write and evaluate `solution.py`.
- **`cwm`**: collect real observations, validate a code world model, then dream-check
  exploration controllers before running them in a fresh real environment.

Select CWM with `protocol=cwm features=none`. Its options are in
[`conf/protocol/cwm.yaml`](../src/regact/conf/protocol/cwm.yaml). It requires
`problem.lifecycle=multi_instance` and OS-isolated code workers. Its three tools
are UpdateCodeWorldModel, PlanInCWM and SubmitExplorationController. It does not
register SubmitSolution or ExitTask and does not perform a final policy re-score.

“Policy search” means that the code agent develops and evaluates a policy in
`solution.py`. “Controller” remains the name of that Python artifact and its
existing `controller.*` evaluation settings. An experiment protocol is different
from `agent.args.tool_protocol`, which selects the syntax used to invoke tools.

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
provides tools, hooks, a reminder, and a stop reason. The shared loop does not
interpret submission counts, scores, or CWM phases.

The order at a turn boundary remains: interruption, protocol outcome, turn limit,
tool-call limit, walltime limit. Mid-turn cooperative stops wait for all observed
tool calls to return. The exit reason is saved before teardown hooks run; hook
failures are logged as before. Backend retries and the no-tool-progress breaker
retain their existing behavior.

## Preserving existing experiments

Existing launch commands require no changes. The explicit selection is:

```sh
python -m regact.run_exp protocol=policy_search agent=codex problem=minigrid
```

All `controller.*` names/defaults, prompts, scaffolded files, SubmitSolution and
ExitTask behavior, evaluation, result paths, and final re-score remain unchanged.
Run configuration artifacts additionally record `protocol: {name: policy_search}`.
The executor and submission implementation are reused unchanged. Alan's remote
scripted mode skips its unused token-escalation setting for compatibility with
newer Alan versions; normal model-run handling is unchanged.

The feature mechanism is composed **inside policy_search**. It remains an
extension point for additive capabilities, not alternative workflows. The old
`features=cwm` launch is retired and fails with a migration message: use
`protocol=cwm features=none` and `protocol.*` parameters. Legacy implementation
modules are retained for migration tests; v4 does not execute them.

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

`tests/test_protocols.py` checks 120 prompt variants and four workspace layouts
against SHA-256 fingerprints captured from pre-refactor commit
`728aa2079aeaa947cb8cb215a481160995d99d9a`. The matrix covers every tool syntax,
ExitTask on/off, historical legacy CWM scaffold variants, both lifecycle prompt variants and the optional
verbalization hints. Testing single-instance text does not enable its rejected
on-environment evaluation configuration.

The existing integration tests cover evaluation failures, perfect-score stopping,
ExitTask, finalization, retries and waiting for enclosing/parallel tool results.
An independent toy protocol test verifies that the common runner can operate
without controller artifacts or submission semantics and still enforces limits.

`tests/test_cwm_protocol.py` exercises phase boundaries, validation, planning,
fresh dream/real controllers, counterexamples, request deduplication, deadlines,
shutdown, immutable snapshots, filesystem isolation and reconstructed playback.
`scripts/local/cwm_v4_remote.sh` provides a no-token-cost Alan remote walkthrough.
