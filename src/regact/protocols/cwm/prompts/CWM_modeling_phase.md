# CWM Modeling

**Goal:** represent what you have observed and predict its transitions exactly. Inspect recorded observations and transitions, form hypotheses, implement them, and validate. Acceptance checks known evidence; it does not prove that you have understood every situation in the game.

If framework tools are available directly, invoke them by name. Otherwise use the terminal commands shown in these documents.

## 1. Implement these four files

| File in `world_model/` | Required definition | Meaning |
|---|---|---|
| `model_state.py` | `State` | Your compact state representation, preferably a dataclass. |
| `model_parser.py` | `parse(obs) -> State` | Extract that state from a complete observation dictionary. |
| `model_render.py` | `render(state) -> dict` | Reconstruct the complete observation dictionary. |
| `model_transition.py` | `step(state, action) -> State` | Predict the state after one game-format action. |

Use normal workspace imports, e.g. `from world_model.model_state import State` or `from world_model.model_parser import parse`; Regact makes the workspace root importable in scripts and submitted code. Callbacks must be repeatable for identical inputs and must not mutate their inputs. `step` predicts the game; it never contacts the real environment.

The complete observation contains `frame`, `reward`, `is_done`, `available_actions` and `info`. Preserve all of them, including nested metadata, frame shape and list order. JSON dictionary key order is irrelevant. Read data dictionaries rather than inferring all fields from PNGs. `obs["info"]["milestones"]` lists events produced by the action leading to this observation; an empty list means no event. The list is not cumulative.

States may use JSON-compatible values, tuples and instances of classes defined in submitted Python modules. Dictionary keys cannot begin with `@` (reserved encoding). Use compact fields and reusable rules/constants rather than keeping a full grid in every State. Serialized state size includes class/field names; it is not Python source size or process memory.

## 2. Submit for validation

Run `python framework/commands.py UpdateCodeWorldModel` with no arguments. It reads your current `world_model/` code and checks all recorded evidence:

1. **Reconstruction:** `render(parse(obs))` equals the complete `obs`.
2. **Prediction:** for every recorded `(obs, action, next_obs)`, `render(step(parse(obs), action))` equals `next_obs`.
3. **Distinct states:** distinct observations cannot parse to the same State.
4. **Repeatability:** callbacks give the same results on repeated identical inputs.
5. **Compactness:** `sum(serialized State bytes) / sum(serialized observation bytes)` over the unique observations must be strictly below **__SIZE_RATIO__**. This is a ratio of totals, not a separate threshold on each observation.

## 3. Use the result

| `status` | Meaning | Next action |
|---|---|---|
| `Accepted` | All checks finished and passed. | Follow the phase notice into Active Exploration. |
| `Refused` | Checks finished but some failed. | Inspect evidence, revise the CWM, submit again. |
| `Incomplete` | Validation could not finish, e.g. a code error or timeout. | Fix the reported error or expensive computation, then retry. |

`checked` counts processed observations/transitions, not just successful checks. `cwm_version` identifies the accepted frozen code; `dataset_version` identifies the evidence checked. These are identifiers, not quality scores; version numbers may have gaps. A refused/incomplete replacement leaves the previous accepted version unchanged. You may explore only while the current phase is Active Exploration. Editing CWM files or their imported dependencies does not update the accepted CWM. Submit those changes with UpdateCodeWorldModel first; otherwise real exploration is refused before taking a real action.

Counterexamples include evidence IDs and one of these failure types:

- `reconstruction_mismatch`: information was lost or rendered incorrectly.
- `prediction_mismatch`: the predicted successor disagreed with the recorded one.
- `parser_collision`: two different observations share a State.
- `compression_ratio`: the aggregate representation is too large; size feedback includes the smallest/largest state and their observation IDs to help inspect it.
- `non_deterministic_model`: identical callback inputs produced different outputs.

`failures` counts failed checks; one bug can cause several. Examples are limited to __COUNTEREXAMPLES__, with at most __DIFF_ITEMS__ differences per example. `differences_omitted` counts additional differences when there are any. Error text is limited to __ERROR_CHARS__ characters, retaining its beginning and end.

Use `load_observations`, `load_transitions` or `load_diagnostic` from `data_api` with the returned IDs. `save_image(diagnostic_id=..., which="observed", path=...)` and `which="predicted"` help compare images, when the diagnostic contains them. Size errors and code exceptions may have no image; inspect their structured data.

## Code execution rules

Develop and inspect data freely within your workspace. Submitted __CALLBACK_KINDS__ callbacks run separately without access to the experience database, workspace, real environment or network. Pass information through their inputs and Python constants; do not call `data_api` inside submitted callbacks.

The framework saves the submitted Python files and their static local imports. External data files and dynamic local imports are not included. Use ordinary Python imports for shared code (including the provided action helpers). A submission may contain at most 256 Python files and 10 MiB of Python source.

One callback request has a __CALL_SECONDS__ second limit, including communication and serialization. This applies to imports/startup, CWM `parse`/`render`/`step`, controller creation and `act`/`is_done`__GOAL_CALLBACK_LIMIT__. It is not a timeout on your Bash commands. Each submitted-code process has a __MEMORY_MB__ MiB memory limit. An `unlimited` value disables that particular cap; whole-task limits still apply.

Budget errors name the effective limit and value. They usually stop one operation, not the whole task. A separate phase notice tells you when to change phase. `task_stop` means the task ended. `history_complete=false` indicates an environment or recording failure: stored counts may omit actions that actually happened.

__LOCAL_HELPERS__
