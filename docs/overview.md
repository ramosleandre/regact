# Overview

regact drives a **code-writing agent** that plays an unknown **game** by writing and
submitting a controller for evaluation in the default **`policy_search`** protocol.
The agent, problem and experiment protocol are separate extension points. Optional
features currently extend policy search. See [Experiment protocols](protocols.md)
for the boundary and the CWM v4 migration status.

## Extension points

| Seam | What it is | Config group | Registry |
|---|---|---|---|
| **Agent** | who writes the code (Claude, codex, Alan, scripted) | `agent=` | closed enum `AgentName` + `build_agent` |
| **Environment** (problem) | the game the agent plays (ARC-AGI-3, MiniGrid) | `problem=` | `register_problem` (open, string-keyed) |
| **Protocol** | the experiment workflow, tools and completion rules | `protocol=` | `register_protocol` (open, string-keyed) |
| **Feature** | optional capabilities added to policy search | `features=` | `register_feature` (open, string-keyed) |

Agents use a *closed* registry: adding one requires editing the enum and factory.
Problems, protocols and features have *open*, string-keyed registries, but their
registration code must run before selection. Built-ins are loaded by their factories;
there is no automatic plugin discovery. Built-ins load lazily so
importing the core does not require every game library or agent SDK.

## How a policy-search run flows

1. **Compose** — [`run_exp.py`](../src/regact/run_exp.py) lets Hydra assemble
   `agent` + `problem` + `protocol` + `controller` + `features` + run-level fields into one typed `RunConfig`.
2. **Schedule** — `run_experiment` resolves the task list and creates a timestamped run
   dir, then the [`Scheduler`](../src/regact/orchestration/scheduler.py) runs each task
   (sequentially, or `parallel_workers` at a time).
3. **Run one task** — [`run_task`](../src/regact/orchestration/task.py) builds the env
   session behind an HTTP boundary and asks the selected protocol for its workspace
   files, tools, hooks and prompt. The shared loop runs until the protocol reports
   completion or a runtime stopping condition applies.
4. **Score** — the agent's submitted code is evaluated by rolling episodes on the env;
   results land under the task's `workdir/submissions/`. Videos, when enabled, are
   recorded during final evaluation under `workdir/submissions/final/`.

## The anti-cheat spine

The agent interacts with the environment **over localhost HTTP**. With `sandbox=true`,
the OS sandbox hides the game source, preventing the agent from bypassing exploration
by reading the implementation. Network isolation is enabled by default under sandboxing;
it restricts access to the sanctioned environment/model connections, including allowed
LLM hosts for cloud agents. These controls address access to hidden game information.
See **[Sandboxing](sandboxing.md)** for the configuration and enforcement details.

## Where to go next

- Pick and configure a backend → **[Agents](agents.md)**
- Pick and configure a game → **[Environments](environments.md)**
- Understand or add a workflow → **[Experiment protocols](protocols.md)**
- Add capabilities to policy search → **[Features](features.md)**
- Launch runs and inspect them → **[Experiments](experiments.md)**
