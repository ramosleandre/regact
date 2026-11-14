# Modeling Phase

**Goal:** represent what you have observed and predict its transitions exactly. Inspect recorded observations and transitions, form hypotheses, implement them, and validate. Acceptance checks known evidence; it does not prove that you have understood every situation in the game.

If framework tools are available directly, invoke them by name. Otherwise use the terminal commands shown in these documents.

## 1. Implement these four files

| File in `world_model/` | Required definition | Meaning |
|---|---|---|
| `model_state.py` | `State` | Your compact state representation, preferably a dataclass. |
| `model_initial_state.py` | `get_initial_state(obs) -> State` | Build the State from the first observation of an episode. |
| `model_render.py` | `render(state) -> dict` | Reconstruct the complete observation dictionary. |
| `model_transition.py` | `step(state, action) -> State` | Predict the State after one game-format action. It must carry everything later observations depend on, including what the screen does not show. |

Use normal workspace imports, e.g. `from world_model.model_state import State` or `from world_model.model_transition import step`; the framework makes the workspace root importable in scripts and submitted code. Callbacks must be repeatable for identical inputs and must not mutate their inputs. `step` predicts the game; it never contacts the real environment.

A world model computes observations from the game's rules. Do not copy recorded observations or transitions into your code and look them up (by action history, by observation id, by frame): such a table predicts nothing new and is not allowed. The game's static content you read from the data (level layouts, sprites, colours) may be written into code.

The complete observation contains `frame`, `reward`, `is_done`, `available_actions` and `info`. Preserve all of them, including nested metadata, frame shape and list order. JSON dictionary key order is irrelevant. Read data dictionaries rather than inferring all fields from PNGs. `obs["info"]["milestones"]` lists events produced by the action leading to this observation; an empty list means no event. The list is not cumulative.

## How the State is carried

An *episode* begins at an environment start, or at a framework reset command when the game has one, and runs until the next one. `get_initial_state` builds the State from the episode's first observation; from there the State evolves through `step`, one recorded action at a time. Completing a level, and a reset that is a game action, are ordinary actions: `step` predicts them, and the State carries through them. Later observations can depend on things the screen does not show, and `step` carries them.

Example of hidden state: a door that opens after its switch has been pressed three times, with nothing on screen counting the presses. One frame cannot tell how many presses are left. Keep a `presses` field in the State: `get_initial_state` sets it to 0, `step` adds 1 on every press and resets it when a level completes, and `render` draws the door from it.

Your controller receives the State carried this way. A hidden field that is wrong but has not yet shown up on screen misleads the controller until a contradiction reveals it.

States may use JSON-compatible values, tuples and instances of classes defined in submitted Python modules. Dictionary keys cannot begin with `@` (reserved encoding). Use compact fields and reusable rules/constants rather than keeping a full grid in every State. The size counted is the number of UTF-8 bytes of the State as compact JSON with sorted keys (no spaces), where a tuple becomes `{"@tuple":[...]}` and an instance of your classes becomes `{"@class":"module:Qualname","fields":{...}}`, so class and field names count; observations are measured the same way. It is not Python source size or process memory.

## 2. Submit for validation

Run `python framework/commands.py UpdateCodeWorldModel` with no arguments. It reads your current `world_model/` code and replays every recorded episode in time order:

1. **Episode start:** `state = get_initial_state(first observation)`, and `render(state)` must equal that observation.
2. **Every recorded action:** `state = step(state, action)`, and `render(state)` must equal the next recorded observation. An episode stops at its first divergence: the State after it is unreliable.
3. **Repeatability:** callbacks give the same results on repeated identical inputs (checked on a sample of calls).
4. **Compactness:** `sum(serialized State bytes) / sum(serialized observation bytes)` must be strictly below **__SIZE_RATIO__**, and so must each single State divided by the larger of its own observation and the average observation. States that grow along an episode (visited sets, action logs) fail.__STATE_ALLOWANCE__

Submitting is the cheapest way to check your CWM against everything recorded: it costs one tool call and no game action, and a refusal costs nothing else. Prefer it to replaying recorded data with your own script.

## 3. Use the result

| `status` | Meaning | Next action |
|---|---|---|
| `Accepted` | All checks finished and passed. | Its `phase_change` field moves you to the Acting Phase. |
| `Refused` | Checks finished but some failed. | Inspect evidence, revise the CWM, submit again. |
| `Incomplete` | Validation could not finish, e.g. a code error or timeout. | Fix the reported error or expensive computation, then retry. |

`checked` counts the episodes and steps processed, not just successful checks. `cwm_version` identifies the accepted frozen code. It is an identifier, not a quality score: it counts accepted CWMs (1, 2, 3, ...). A refused/incomplete replacement leaves the previous accepted version unchanged. Controllers run only in the Acting Phase. Editing CWM files or their imported dependencies does not update the accepted CWM. Submit those changes with UpdateCodeWorldModel first; otherwise RunController is refused before taking an action.

Counterexamples include evidence IDs and one of these failure types:

- `reconstruction_mismatch`: `render(get_initial_state(obs))` does not reproduce an episode's first observation.
- `prediction_mismatch`: the State carried by `step` stopped matching a recorded observation. It names the `episode_id` and `step` (actions applied in that episode). The real mistake can be earlier, when a hidden field went wrong before it showed on screen.
- `compression_ratio`: States are too large on average or for one State; size feedback includes the smallest/largest State and their observation IDs.
- `non_deterministic_model`: identical callback inputs produced different outputs.

`failures` counts failed checks; one bug can cause several. Examples are limited to __COUNTEREXAMPLES__. They have the same fields as a contradiction during a controller call, described in "Reading a contradiction" in `docs/acting_phase.md`. Error text is limited to __ERROR_CHARS__ characters, retaining its beginning and end.

To understand a counterexample, start with its `details` call, `data_api.describe_diagnostic(diagnostic_id)`: the differences, your model's State before the step and the State it predicted, all bounded. `data_api.load_diagnostic(diagnostic_id)` returns the whole record and can be very large. Use `load_observations` or `load_transitions` from `data_api` with the returned IDs, and `data_api.list_episodes()` and `data_api.load_history(episode_id)` to see what happened in order. Only when a hidden field may have gone wrong before the failing step, replay the episode locally: `observations, actions = data_api.load_history(episode_id)`, `state = get_initial_state(observations[0])`, then `state = step(state, action)` for each action, checking the field you suspect.__DIAGNOSTIC_IMAGES__

## Code execution rules

Develop and inspect data freely within your workspace. Submitted __CALLBACK_KINDS__ callbacks run separately without access to the experience database, workspace, real environment or network. Pass information through their inputs and Python constants; do not call `data_api` inside submitted callbacks.

The framework saves the submitted Python files and their static local imports. External data files and dynamic local imports are not included. Use ordinary Python imports for shared code (including the provided action helpers). A submission may contain at most 256 Python files and 10 MiB of Python source.

One callback request has a __CALL_SECONDS__ second limit, including communication and serialization. This applies to imports/startup, CWM `get_initial_state`/`render`/`step`, controller creation and `act`/`is_done`__GOAL_CALLBACK_LIMIT__. It is not a timeout on your Bash commands. Each submitted-code process has a __MEMORY_MB__ MiB memory limit. An `unlimited` value disables that particular cap; whole-task limits still apply.

Budget errors name the effective limit and value. They usually stop one operation, not the whole task. When a command changes the phase, its result has a `phase_change` field, such as `"Modeling Phase -> Acting Phase"`. `task_stop` means the task ended. `history_complete=false` indicates an environment or recording failure: stored counts may omit actions that actually happened.

__LOCAL_HELPERS__
