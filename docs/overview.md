# Overview

regact drives a **code-writing agent** that plays an unknown **game** by writing and
submitting code. Everything is built around three pluggable seams, so an experiment is
"pick one of each and run".

## The three seams

| Seam | What it is | Config group | Registry |
|---|---|---|---|
| **Agent** | who writes the code (Claude, codex, Alan, scripted) | `agent=` | closed enum `AgentName` + `build_agent` |
| **Environment** (problem) | the game the agent plays (ARC-AGI-3, MiniGrid) | `problem=` | `register_problem` (open, string-keyed) |
| **Feature** | what the agent builds and how it's scored | `features=` | `register_feature` (open, string-keyed) |

An agent is a *closed* set (adding one edits an enum); problems and features are *open*
(a new module self-registers under a name, no core edit). All three import their built-ins
lazily, so a base install never pulls a game library or an agent SDK.

## How a run flows

1. **Compose** — [`run_exp.py`](../src/regact/run_exp.py) lets Hydra assemble
   `agent` + `problem` + `features` + run-level fields into one typed `RunConfig`.
2. **Schedule** — `run_experiment` resolves the task list and creates a timestamped run
   dir, then the [`Scheduler`](../src/regact/orchestration/scheduler.py) runs each task
   (sequentially, or `parallel_workers` at a time).
3. **Run one task** — [`run_task`](../src/regact/orchestration/task.py) builds the env
   session behind an HTTP boundary, bootstraps the agent's workdir, wires the features'
   tools + hooks, builds the prompt, and drives the keep-alive loop until the agent
   submits and exits (or hits a limit).
4. **Score** — the agent's submitted code is evaluated by rolling episodes on the env;
   results and a video land under the task's `workdir/submissions/`.

## The anti-cheat spine

The agent reaches the environment **only over localhost HTTP** and never imports the
game — so the score can't come from reading the answer. On a scored run, an OS sandbox
additionally makes the game files *absent* from the agent's filesystem and blocks the
internet, while keeping the sanctioned localhost path open. This is the whole point of
the framework: the number measures understanding, not memorization. See
**[Sandboxing](sandboxing.md)**.

## Where to go next

- Pick and configure a backend → **[Agents](agents.md)**
- Pick and configure a game → **[Environments](environments.md)**
- Pick and configure what the agent builds → **[Features](features.md)**
- Launch runs and inspect them → **[Experiments](experiments.md)**
