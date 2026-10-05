# CWM Modeling

**Goal:** represent what you have observed and predict its transitions exactly. Inspect recorded observations and transitions, form hypotheses, implement them, and validate. Acceptance checks known evidence; it does not prove that you have understood every situation in the game.

If framework tools are available directly, invoke them by name. Otherwise use the terminal commands shown in these documents.

## 1. Implement these four files

| File in `world_model/` | Required definition | Meaning |
|---|---|---|
| `model_state.py` | `State` | Your compact state representation, preferably a dataclass. |
| `model_initial_state.py` | `get_initial_state(obs) -> State` | Build the State from the first observation of an episode: the game's start, or the start of any level after a reset. |
| `model_render.py` | `render(state) -> dict` | Reconstruct the complete observation dictionary. |
| `model_transition.py` | `step(state, action) -> State` | Predict the State after one game-format action. It must carry everything later observations depend on, including what the screen does not show. |

Use normal workspace imports, e.g. `from world_model.model_state import State` or `from world_model.model_transition import step`; Regact makes the workspace root importable in scripts and submitted code. Callbacks must be repeatable for identical inputs and must not mutate their inputs. `step` predicts the game; it never contacts the real environment.

The complete observation contains `frame`, `reward`, `is_done`, `available_actions` and `info`. Preserve all of them, including nested metadata, frame shape and list order. JSON dictionary key order is irrelevant. Read data dictionaries rather than inferring all fields from PNGs. `obs["info"]["milestones"]` lists events produced by the action leading to this observation; an empty list means no event. The list is not cumulative.

## How the State is carried

An *episode* begins at an environment start or reset and runs until the next reset. `get_initial_state` builds the State from the episode's first observation; from there the State evolves through `step`, one recorded action at a time. A level reset restarts the current level, so that first observation can be the start of any level you have reached: `get_initial_state` must recognise which one and return its starting State. Completing a level is an ordinary action: `step` produces the next level's starting State. The first observation of an episode determines its State; the start of a level can look different from one reset to the next, so read it from the observation. Later observations can depend on things the screen does not show, and `step` carries them.

Example of hidden state: a budget bar with 64 cells for a budget of 75 clicks, where a click blocked by a wall still spends budget. One frame cannot tell how many clicks are left. Keep a `clicks` field in the State: `get_initial_state` sets it from the full bar, `step` adds 1 on every click (blocked ones included) and resets it when a level completes, and `render` draws the bar from it.

Your controller receives the State carried this way. A hidden field that is wrong but has not yet shown up on screen misleads the controller until a contradiction reveals it.

States may use JSON-compatible values, tuples and instances of classes defined in submitted Python modules. Dictionary keys cannot begin with `@` (reserved encoding). Use compact fields and reusable rules/constants rather than keeping a full grid in every State. Serialized state size includes class/field names; it is not Python source size or process memory.

## 2. Submit for validation

Run `python framework/commands.py UpdateCodeWorldModel` with no arguments. It reads your current `world_model/` code and replays every recorded episode in time order:

1. **Episode start:** `state = get_initial_state(first observation)`, and `render(state)` must equal that observation.
2. **Every recorded action:** `state = step(state, action)`, and `render(state)` must equal the next recorded observation. An episode stops at its first divergence: the State after it is unreliable.
3. **Repeatability:** callbacks give the same results on repeated identical inputs (checked on a sample of calls).
4. **Compactness:** `sum(serialized State bytes) / sum(serialized observation bytes)` must be strictly below **__SIZE_RATIO__**, and so must the ratio of the single largest State. States that grow along an episode (visited sets, action logs) fail.

## 3. Use the result

| `status` | Meaning | Next action |
|---|---|---|
| `Accepted` | All checks finished and passed. | Its `phase_change` field moves you to Active Exploration. |
| `Refused` | Checks finished but some failed. | Inspect evidence, revise the CWM, submit again. |
| `Incomplete` | Validation could not finish, e.g. a code error or timeout. | Fix the reported error or expensive computation, then retry. |

`checked` counts the episodes and steps processed, not just successful checks. `cwm_version` identifies the accepted frozen code; `dataset_version` identifies the evidence checked. These are identifiers, not quality scores: cwm_version counts accepted CWMs (1, 2, 3, ...). A refused/incomplete replacement leaves the previous accepted version unchanged. You may explore only while the current phase is Active Exploration. Editing CWM files or their imported dependencies does not update the accepted CWM. Submit those changes with UpdateCodeWorldModel first; otherwise real exploration is refused before taking a real action.

Counterexamples include evidence IDs and one of these failure types:

- `reconstruction_mismatch`: `render(get_initial_state(obs))` does not reproduce an episode's first observation.
- `prediction_mismatch`: the State carried by `step` stopped matching a recorded observation. It names the `episode_id` and `step` (actions applied in that episode). The real mistake can be earlier, when a hidden field went wrong before it showed on screen.
- `compression_ratio`: States are too large on average or for one State; size feedback includes the smallest/largest State and their observation IDs.
- `non_deterministic_model`: identical callback inputs produced different outputs.

`failures` counts failed checks; one bug can cause several. Examples are limited to __COUNTEREXAMPLES__, with at most __DIFF_ITEMS__ differences per example. `differences_omitted` counts additional differences when there are any. Error text is limited to __ERROR_CHARS__ characters, retaining its beginning and end.

Use `load_observations`, `load_transitions` or `load_diagnostic` from `data_api` with the returned IDs. To see what happened in order, use `data_api.list_episodes()` and `data_api.load_history(episode_id)`. To reproduce a counterexample locally: `observations, actions = data_api.load_history(episode_id)`, `state = get_initial_state(observations[0])`, then for each action `state = step(state, action)` and compare `render(state)` with the next observation.__DIAGNOSTIC_IMAGES__

## Code execution rules

Develop and inspect data freely within your workspace. Submitted __CALLBACK_KINDS__ callbacks run separately without access to the experience database, workspace, real environment or network. Pass information through their inputs and Python constants; do not call `data_api` inside submitted callbacks.

The framework saves the submitted Python files and their static local imports. External data files and dynamic local imports are not included. Use ordinary Python imports for shared code (including the provided action helpers). A submission may contain at most 256 Python files and 10 MiB of Python source.

One callback request has a __CALL_SECONDS__ second limit, including communication and serialization. This applies to imports/startup, CWM `get_initial_state`/`render`/`step`, controller creation and `act`/`is_done`__GOAL_CALLBACK_LIMIT__. It is not a timeout on your Bash commands. Each submitted-code process has a __MEMORY_MB__ MiB memory limit. An `unlimited` value disables that particular cap; whole-task limits still apply.

Budget errors name the effective limit and value. They usually stop one operation, not the whole task. When a command changes the phase, its result has a `phase_change` field: `from`, `to`, and `next_step`. `task_stop` means the task ended. `history_complete=false` indicates an environment or recording failure: stored counts may omit actions that actually happened.

__LOCAL_HELPERS__
