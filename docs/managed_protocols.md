# Managed execution: vanilla and CWM

In `vanilla` and `cwm`, Regact owns the live environment. The agent reads recorded experience and submits Python controllers through `RunController`. It cannot step or reset the environment through a separate client.

This guide covers their shared behavior and all shared protocol parameters. [CWM v5](cwm.md) explains the additional modeling requirements. [Protocols](protocols.md) compares these workflows with policy search.

## Run, call and episode

- An **experiment run** contains one or more tasks, possibly with repeated attempts.
- A **task attempt** has its own agent session, workspace, dataset and budgets.
- A **controller call** is one `RunController` invocation. It constructs a fresh controller object.
- An **episode** starts at an environment reset. In `single_instance`, it can contain initial collection and several controller calls.

The coding agent's tool calls and real environment actions are different counts: one Bash call invoking `RunController` can execute many game actions. An outer Regact **turn** is one send cycle to the backend; a CLI backend can perform many tool calls within it.

## A task, step by step

### 1. Create the environment and collect initial data

Regact records the reset observation, then runs a random policy before starting the agent. It targets `protocol.n_unique_observations_in_initial_collection=20` distinct complete observations, counting the reset observation. It stops earlier on `protocol.max_actions_per_initial_collection`, `protocol.max_seconds_per_initial_collection`, a task/episode limit, no available actions, or environment termination.

The action sampler uses `problem.seed` (0 when unspecified). Environment seeding still follows the problem's own `problem.seed` behavior; a seeded action sampler alone does not guarantee identical environments. Collection never silently resets a terminated game. Its actions count against task and episode action budgets. If collection solves the game, the task can finish before the agent acts.

Initial experience and later experience use the same dataset interface. There is no agent-facing phase 0. CWM starts in **CWM Modeling**; vanilla can immediately submit a controller.

### 2. Inspect the dataset and develop code

The agent starts with `controller.py`, `framework/commands.py`, `framework/data_api.py`, `framework/action_list_controller.py`, and the configured game helpers. CWM adds its world-model files and phase guides. Files are relative to the agent's workspace, not the Regact checkout.

For example, a workspace script can read the next controller's starting observation:

```python
from framework import data_api

summary = data_api.summary()
obs_id = summary["controller_start_observation_id"]
obs = data_api.load_observations([obs_id])[0]
print(obs["available_actions"])
data_api.save_image("current.png", observation_id=obs_id)
```

A vision agent then opens `current.png` with its own image-reading tool; a text-only agent is told to analyse the arrays instead. Dataset reads and image creation take no game actions. The default `first_obs_in_prompt=false` keeps the initial grid out of the first message.

### 3. Implement a controller

`controller.py` needs a module docstring describing the intended experiment and a `get_controller()` factory. A minimal vanilla example is:

```python
"""Test one available action from the current observation."""

class Controller:
    def __init__(self):
        self.n_actions = 0

    def is_done(self, obs):
        return self.n_actions >= 1

    def act(self, obs):
        self.n_actions += 1
        return obs["available_actions"][0]

def get_controller():
    return Controller()
```

This simple example assumes an available action is directly executable; parameterized actions need the game's action format/helper. In CWM, `act` and `is_done` receive the accepted CWM's **State**, not the observation dictionary.

`is_done` is checked before the next action. Returning `False` is valid; environment termination and budgets still stop execution. An optional `objective_reached(input) -> bool` reports a chosen subgoal independently of stopping. It is not required and does not establish full-game success. The supplied `ExplorationControllerFromListActions` stops when its list is exhausted; exhaustion alone is not goal attainment.

Controllers can keep private memory during a call. Every later `RunController` constructs a new object, even when it continues the same physical episode.

### 4. Run against the real environment

```bash
python framework/commands.py RunController
```

The command takes no arguments and reads `controller.py`. Regact snapshots its Python code and static local imports, constructs an isolated controller, and executes actions until it stops. CWM also requires an accepted, unchanged CWM and checks predictions around each real action; see [the CWM execution sequence](cwm.md#active-exploration).

| `problem.lifecycle` | Start of each `RunController` |
|---|---|
| `single_instance` | Current live observation, including where initial collection or the previous controller stopped |
| `multi_instance` | Fresh environment reset; CWM checks that it matches the original initial observation |

Single-instance controller completion, errors and CWM contradictions do not implicitly reset the environment. Multi-instance CWM therefore needs a reproducible initial observation; a varying reset can end the task with `initial_observation_mismatch`.

### 5. Inspect feedback, then continue or reset

Command results are indented JSON, with separate phase notices when relevant. They identify why the call stopped, actions taken, game metrics, evidence IDs, and any error/diagnostic. `goal_achieved` refers to the controller's optional subgoal; the problem's success predicate determines full-game completion.

`observation_ids` and `transition_ids` identify the distinct records encountered by the call, including reused records and its starting observation. Ranges such as `[4:9]` are **inclusive** and can be passed directly to the data API. They are not a chronological path; logs retain ordered occurrences separately for playback.

`actual_novel_observations` counts new complete observations. Novelty is a signal, not a requirement to execute the controller or proof of useful progress. `new_milestones` highlights first-time real events. Within an observation, `info["milestones"]` describes the preceding action's events, not cumulative achievement.

When enabled, previews are written to `tmp/images/obs_id_<ID>.png`. At most `protocol.n_tmp_images_saved_per_exploration` images are selected from the first and last distinct observations encountered. The folder is emptied on every new `RunController`, including refused calls. The underlying dataset is retained.

### 6. End the call or end the task

| Event | Effect |
|---|---|
| Controller `is_done(...)` is true | Finish this call |
| Observation `is_done=True` | Finish this call; an unsuccessful episode can be inspected and reset |
| Per-call action/time cap | Finish this call; the agent can adapt its next controller |
| Episode action cap | No further steps in that episode; a reset renews that action allowance |
| Controller error, including an invalid action | Finish the call, report the error and worst performance metrics; keep collected evidence |
| CWM contradiction | Finish the call and return to CWM Modeling |
| Full game solved according to problem metrics | End the task |
| Task tool/action/turn/wall-time limit or interruption | End the task |
| Unrecoverable framework/storage failure | End the task with an explicit failure; history may be incomplete |

ARC's `info.state="GAME_OVER"` is game metadata. The generic framework uses `is_done` and problem metrics rather than branching on that string. There is no `ExitTask` command or independent final policy re-score in these two protocols.

## Explicit resets

| Command | Offline ARC | MiniGrid |
|---|---|---|
| `ResetLevel` | Restart the current level, preserving completed levels | Not available |
| `ResetEnvironment` | Restart from level 1 | Restart the environment |

Invoke them with `python framework/commands.py ResetLevel` or `ResetEnvironment`, without arguments. Each successful explicit reset consumes **one task action**, starts a recorded episode, and renews the episode action allowance. It does not clear the dataset or replenish task budgets. Reset boundaries are not ordinary CWM transition examples.

CWM permits resets in either phase. A new reset observation requires validation; an already validated one does not cause redundant validation. ARC `RESET` is rejected as a controller action: use the explicit commands. Separate ARC reset scopes currently depend on its offline/local engine.

## Dataset API

The generated `framework/data_api.py` contains full docstrings and examples. No function below steps or resets the environment.

| Function | Returns / purpose |
|---|---|
| `summary()` | Current phase, observation/episode IDs, counts, reset count and recorded milestones |
| `list_observation_ids(after_id=0, limit=100)` | Page of unique observation IDs, ascending |
| `list_transition_ids(after_id=0, limit=100)` | Page of unique transition IDs, ascending |
| `load_observations(ids)` | Complete observation dictionaries in the requested order |
| `load_transitions(ids)` | `transition_id`, `before_obs_id`, `after_obs_id`, `action`, `o`, `o_next` |
| `load_diagnostic(diagnostic_id)` | Complete evidence for a reported failed check; content depends on the diagnostic |
| `save_image(path, ...)` | Write a PNG, print its source/path, return `None` |

For `save_image`, select exactly one source: `observation_id`; `transition_id` with `which="before"` or `"after"` (default); or `diagnostic_id` with `which="predicted"` or `"observed"` (default). Not every diagnostic contains images. The parent directory must exist. An image shows `frame`; inspect the observation dictionary for reward, terminal status and metadata.

### Summary fields

| Field | Meaning |
|---|---|
| `phase` | Current workflow phase |
| `initial_observation_id` | First observation recorded in this task |
| `current_observation_id` | Live environment's current observation |
| `controller_start_observation_id` | Current ID in single-instance mode; original initial ID in multi-instance mode |
| `episode_id` | Current recorded episode |
| `reset_actions` | Completed explicit reset requests |
| `n_unique_observations` | Distinct complete observation dictionaries |
| `n_total_observations` | All recorded observation occurrences: episode starts plus successors of real steps |
| `n_unique_transitions` | Distinct `(observation, action, successor)` triples |
| `n_total_transitions` | All recorded real steps, including repetitions |
| `n_started_episodes` | Episodes begun, including zero-step episodes |
| `milestones` (when nonempty) | First occurrences with kind and evidence IDs |

IDs identify deduplicated data, not positions in a trajectory. Simulations never enter this real dataset. `dataset_version` is retained in validation/logging evidence; it is deliberately omitted from the agent's ordinary summary.

Pagination defaults to the configured `protocol.data_api.max_items`. Request the next page with `after_id=page[-1]` until the page is empty. `limit=None` requests all remaining IDs only when that cap is `null`. Bulk byte limits produce an error rather than truncated observations: reduce the batch size. A single record, diagnostic or image remains readable even above the bulk byte cap.

## Shared protocol parameters

These are repository defaults from [vanilla.yaml](../src/regact/conf/protocol/vanilla.yaml) and [cwm.yaml](../src/regact/conf/protocol/cwm.yaml). Launchers can override them. Every Regact `max_*` option accepts `null` to disable that particular cap; other caps still apply.

| Full parameter | Default | Scope |
|---|---|---|
| `protocol.n_unique_observations_in_initial_collection` | `20` | Initial collection target, including reset observation |
| `protocol.max_actions_per_initial_collection` | `1000` | Initial random steps |
| `protocol.max_seconds_per_initial_collection` | `30` | Initial collection wall time |
| `protocol.max_actions_per_exploration` | `2500` | Real actions in one `RunController`; the name remains for compatibility |
| `protocol.n_tmp_images_saved_per_exploration` | `8` | Maximum automatic previews per call; `0` disables them |
| `protocol.execution.max_seconds_per_call` | `5` | One isolated callback/startup, including communication and serialization |
| `protocol.execution.max_seconds_per_controller_call` | `90` | One `RunController` execution, including submitted-code startup and CWM/controller computation; below the agents' 120 s shell timeout |
| `protocol.execution.max_memory_mb` | `512` | MiB per isolated submitted-code process |
| `protocol.feedback.max_counterexamples` | `5` | Counterexamples included in CWM feedback; shared schema, normally unused by vanilla |
| `protocol.feedback.max_diff_items` | `6` | Structural differences displayed per comparison |
| `protocol.feedback.max_error_chars` | `1000` | Error text length; truncation preserves beginning and end |
| `protocol.data_api.max_items` | `100` | Page/batch item cap |
| `protocol.data_api.max_response_bytes` | `2097152` | Serialized bulk response cap (2 MiB); single-record exceptions described above |

The callback cap applies to imports/startup, controller construction, `act`/`is_done`, optional controller callbacks, and, in CWM, `parse`/`render`/`step` and planner goal callbacks. It is separate from shell-tool timeouts.

Each `RunController` gets a fresh time allowance. **There is no accumulated time cap per physical episode.** `protocol.execution.max_seconds_per_episode` was removed. Initial collection, validation and planning have their own limits. Budget errors identify the effective parameter and its value.

### Task-level settings

| Full parameter | Default | Meaning |
|---|---|---|
| `limits.max_tool_calls` | `null` | Agent tool calls per task; one framework command can contain many real steps |
| `limits.max_turns_per_task` | `350` | Outer Regact send cycles, not backend-internal messages |
| `limits.max_seconds_per_task` | `null` | Whole-task wall time, including preparation and agent work |
| `limits.max_actions_per_task` | `null` | Real steps plus explicit resets across the task |
| `limits.max_actions_per_episode` | `null` | Real steps since the latest reset |
| `limits.max_consecutive_no_tool_turns` | `0` | Stop after this many consecutive outer turns without tools; `0`/`null` disable it |
| `first_obs_in_prompt` | `false` | Add the first observation to the initial agent message |
| `flagging_warning_cap` | `3` | Maximum delivered suspicious-command warnings; `0` hides warnings, not recorded flags |

The tool-call limit uses completed tool results: cooperative stopping waits for outstanding observed calls, including enclosing framework operations. A task wall-time limit or explicit forced interruption can still terminate work.

## Code isolation and practical limits

Submitted code runs without the experience database, agent workspace, live environment, game implementation or network. Use the dataset while developing, and pass runtime information through callback inputs and imported Python code/constants. Normal imports such as `from world_model.model_state import State` work from the workspace root and frozen submissions.

Snapshots include Python source and static local imports, up to 256 files and 10 MiB. External data files and dynamic local imports are not automatically included. Internal submitted-code requests/replies also have a 16 MiB transport cap. These are implementation bounds, distinct from configurable dataset query limits.

Worker isolation is required even if `sandbox=false` disables the coding agent's sandbox. Local development scripts have the agent's permissions, not the submitted worker's permissions. See [Sandboxing](sandboxing.md).

Game-engine `step`/`reset` calls run on the trusted side and are not forcibly interruptible like submitted callbacks. Recording cannot be atomic with an external game action: a process/disk failure after an action can leave incomplete history. There is no automatic crash recovery/resume or total dataset disk quota. Deduplication and on-demand playback reduce storage, but unique observations and saved code versions can still grow.

For output paths, conversation playback and benchmark interpretation, see [Experiments](experiments.md).
