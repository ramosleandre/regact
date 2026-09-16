# Overview

regact drives a **code-writing agent** that plays an unknown **game** by writing and
submitting a controller for evaluation. Three extension interfaces let you select an
agent, a problem, and optional features. The controller is always present, and the
runner requires a lifecycle compatible with its evaluation workflow.

## The three seams

| Seam | What it is | Config group | Registry |
|---|---|---|---|
| **Agent** | who writes the code (Claude, codex, Alan, scripted) | `agent=` | closed enum `AgentName` + `build_agent` |
| **Environment** (problem) | the game the agent plays (ARC-AGI-3, MiniGrid) | `problem=` | `register_problem` (open, string-keyed) |
| **Feature** | optional capabilities added to the controller workflow | `features=` | `register_feature` (open, string-keyed) |

Agents use a *closed* registry: adding one requires editing the enum and factory.
Problems and features have *open*, string-keyed registries, but their modules must be
imported to register them. The built-in extension guides add that import to
`_load_builtins()`; there is no automatic plugin discovery. Built-ins load lazily so
importing the core does not require every game library or agent SDK.

## How a run flows

1. **Compose** — [`run_exp.py`](../src/regact/run_exp.py) lets Hydra assemble
   `agent` + `problem` + `controller` + `features` + run-level fields into one typed `RunConfig`.
2. **Schedule** — `run_experiment` resolves the task list and creates a timestamped run
   dir, then the [`Scheduler`](../src/regact/orchestration/scheduler.py) runs each task
   (sequentially, or `parallel_workers` at a time).
3. **Run one task** — [`run_task`](../src/regact/orchestration/task.py) builds the env
   session behind an HTTP boundary, bootstraps the agent's workdir, wires the features'
   tools + hooks, builds the prompt, and drives the keep-alive loop until the agent
   submits and exits (or hits a limit).
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
- Pick and configure what the agent builds → **[Features](features.md)**
- Launch runs and inspect them → **[Experiments](experiments.md)**
