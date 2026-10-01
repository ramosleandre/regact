# Overview

Regact gives a **code-writing agent** an unknown **game**, a workspace and a controlled way to interact with the game. The agent writes code, the problem defines the game interface, and the protocol defines the experiment workflow.

## Choose a workflow

| Protocol | What the agent develops | How it interacts |
|---|---|---|
| `vanilla` | Observation-based controllers | `RunController` executes code against an environment managed by Regact. |
| `cwm` | A Code World Model and State-based controllers | The same managed execution, with required CWM validation and prediction checks. |
| `policy_search` (default) | A reusable policy in `solution.py` | Direct exploration, then `SubmitSolution` evaluates the policy on fresh episodes. |

Vanilla and CWM share environment management, the dataset, controller execution, resets and logging. Their scores describe observed interaction, without a separate final evaluation. Policy search evaluates submitted policies and re-evaluates the current policy at teardown.

Start with [Protocols](protocols.md) to choose a workflow, [Managed execution](managed_protocols.md) for vanilla and the common CWM behavior, and [CWM v5](cwm.md) for modeling and validation.

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

## How a run flows

1. **Compose** — [`run_exp.py`](../src/regact/run_exp.py) lets Hydra assemble
   `agent` + `problem` + `protocol` + `controller` + `features` + run-level fields into one typed `RunConfig`.
2. **Schedule** — `run_experiment` resolves the task list and creates a timestamped run
   dir, then the [`Scheduler`](../src/regact/orchestration/scheduler.py) runs each task
   (sequentially, or `parallel_workers` at a time).
3. **Run one task** — [`run_task`](../src/regact/orchestration/task.py) builds the env
   session behind an HTTP boundary and asks the selected protocol for its workspace
   files, tools, hooks and prompt. The shared loop runs until the protocol reports
   completion or a runtime stopping condition applies.
4. **Execute the workflow** — policy search evaluates submitted policies. Vanilla runs observation-based controllers. CWM alternates modeling and exploration, checking predictions around each real action. Managed protocols prepare bounded random experience before starting the agent.
5. **Persist and finish** — transcripts and protocol-specific evidence are recorded during execution. The protocol determines completion and finalization; task limits can also stop the run. See [Experiments](experiments.md) for output paths.

## The anti-cheat spine

Two HTTP interfaces separate environment interaction from framework commands. Policy-search agents can reset/step through the environment client. Vanilla/CWM deny direct client access: framework commands execute real actions and provide a read-only dataset API. With `sandbox=true`,
the OS sandbox hides the game source, preventing the agent from bypassing exploration
by reading the implementation. Network isolation is enabled by default under sandboxing;
it restricts access to the sanctioned environment/model connections, including allowed
LLM hosts for cloud agents. These controls address access to hidden game information.
See **[Sandboxing](sandboxing.md)** for the configuration and enforcement details.

Submitted vanilla/CWM code runs in separate, more restricted processes without access to the live environment, dataset, agent workspace or network.

## Code map

| Location | Responsibility |
|---|---|
| [orchestration/](../src/regact/orchestration/) | Scheduling and the common agent loop |
| [protocols/base.py](../src/regact/protocols/base.py) | Protocol/session interfaces |
| [protocols/managed/](../src/regact/protocols/managed/) | Shared vanilla/CWM execution and prompt assembly |
| [protocols/vanilla.py](../src/regact/protocols/vanilla.py) | Observation-based managed protocol |
| [protocols/cwm/](../src/regact/protocols/cwm/) | CWM validation, planning, dataset and isolated execution support |
| [protocols/policy_search.py](../src/regact/protocols/policy_search.py) | Existing policy-search workflow |
| [viz/](../src/regact/viz/) | Saved-run reader and browser viewer |

Some shared storage, feedback and worker utilities currently live under `protocols/cwm/`; their use by vanilla does not make vanilla a CWM subclass.

## Where to go next

- Pick and configure a backend → **[Agents](agents.md)**
- Pick and configure a game → **[Environments](environments.md)**
- Understand or add a workflow → **[Experiment protocols](protocols.md)**
- Add capabilities to policy search → **[Features](features.md)**
- Launch runs and inspect them → **[Experiments](experiments.md)**
