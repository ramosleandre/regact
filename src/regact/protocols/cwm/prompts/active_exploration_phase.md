# Active Exploration

**Goal:** explore new and relevant states, test uncertain game mechanisms and make progress toward solving the game. Choose a concrete experiment. Testing a predicted familiar outcome can expose an incorrect hypothesis.

## 1. Write `controller.py`

Put a short, plain-text description of the experiment in the file's **module docstring** (the first triple-quoted string). Define `get_controller()` returning a new object with:

- `act(state)`: return one action in the game's format, using a CWM State as input.
- `is_done(state)`: return True to stop this exploration before its next action. The default False is valid: the game ending or an action/time limit can stop it. Add an earlier stopping condition when it suits your experiment.

The controller may retain internal memory. Each real exploration starts with a fresh controller instance. Your controller receives the State carried by the accepted CWM: built by `get_initial_state` at the start of the episode, then advanced by `step` and checked against every real observation. __CONTROLLER_KINDS__ cannot query the dataset or the live environment themselves.

Optional `objective_reached(state) -> bool` reports whether your particular goal was achieved. It is independent of `is_done`: stopping because a list ran out does not establish success. You do not need this optional method to submit a controller.

__ACTION_LIST_GUIDANCE__

## 2. Submit the controller

Run `python framework/commands.py RunController` with no arguments. The command uses the accepted, frozen CWM. If its source files or imported dependencies have changed, the command refuses to start: run `UpdateCodeWorldModel` first. Editing only your controller or local test script does not require revalidation unless that file is also imported by the CWM.

The command follows the task lifecycle described in your instructions: it continues the live environment in single-instance mode and starts fresh in multi-instance mode. At each step:

1. Give the current State to the controller and check whether it wants to stop. The first State is `get_initial_state(starting observation)` in multi-instance mode, or the State carried from earlier calls in single-instance mode (after a reset, `get_initial_state` of the reset observation).
2. Obtain an action from `act(state)` and compute the CWM's next State with `step` and its predicted observation with `render`.
3. Execute the action in the real environment, record the transition and compare the actual observation with the prediction. If they match, the predicted State becomes the current State; compactness is checked on it.
4. Stop at the first contradiction, controller completion, game end or limit.

If the CWM cannot compute a prediction, the action is not executed: repair the reported error and revalidate. An incorrect prediction is recorded as a counterexample after executing the action.

## 3. Read the exploration feedback

The result is an indented JSON dictionary. A refusal before starting explains what to fix. Real-execution feedback reports:

- `observation_ids` and `transition_ids`: all recorded evidence encountered in this controller call, including reused IDs and its starting observation. Load these directly with `data_api`; `[4:9]` means IDs 4 through 9, inclusive.
- `real_actions` counts executed actions; `actual_novel_observations` counts newly recorded unique observations. Predictions never enter the real dataset.
- `metrics`: actual game performance. New-milestone notices highlight first-time progress/failure events in real experience. Simulated achievements do not count.
- `goal_achieved`: your controller's optional goal check, not necessarily game completion. Finishing an action list also does not imply its goal was achieved.
- `counterexample` and `differences`: what reality contradicted, described below. A failed check or callback reports an `error` instead, whose cause is also in `message`.

### Reading a contradiction

When a prediction does not match reality, the result's `message` says in one sentence where it happened and what differs. The counterexample gives the details, and `differences` lists what differs between the predicted observation and the observed one.

#### Where it happened

- `episode_id`, `step`: the episode and the number of actions applied in it.
- `from_observation_id`, `action`, `observation_id`, `transition_id`: the step itself; the State before it was carried from `from_observation_id`, and `action` was applied.
- `explained_by_previous_version` (refused `UpdateCodeWorldModel` only): `true` when your previously accepted CWM predicted this step correctly, so the latest change broke it; `false` when no accepted version has explained it yet.
- `diagnostic_id` and `details`: `details` is the call that describes this diagnostic: the same differences with other limits if you pass them, and your model's State before the step and the State it predicted. `data_api.load_diagnostic(diagnostic_id)` returns the complete record, including both full observations; it can be very large.__DIAGNOSTIC_IMAGE__

#### What differs

`differences` lists every field of the observation that differs (`frame`, `info`, `is_done`, ...). Each field is reported separately and within its own limits, so a large frame difference never hides a difference in another field.

Values are listed by path, for example `{"where": "info.state", "predicted": "NOT_FINISHED", "observed": "GAME_OVER"}`. At most __DIFF_ITEMS__ values are listed per field; `other differences` counts the rest.

A grid, such as one frame layer, is summarized as regions. A region is a group of differing cells that touch, including diagonally. `cells_differing` counts the differing cells of the grid and `cells_total` all its cells. At most __DIFF_REGIONS__ regions are listed, largest first; `other_regions` counts the rest.

Each region gives:

- `where`, in Python slice notation with exclusive ends, row first, then column (on a screen: y, then x):
  - `frame[0][45:50, 34:39]`: the rectangle of rows 45 to 49 and columns 34 to 38 of layer 0;
  - `frame[0][63, 0:64]`: one row; `frame[0][63, 63]`: one cell;
  - `frame[0][0:4, 0] + frame[0][3, 1:4]`: exactly these cells, as a sum of rectangles;
  - `within frame[0][9:45, 27:57]`: an irregular region somewhere inside this rectangle, not filling it.
- `cells`: how many cells of the region differ.
- Either `changes`, the `predicted -> observed` value pairs with their cell counts (at most __DIFF_CHANGES__, then `other changes`), or, for a small region holding several changes or an irregular shape, the area itself: `predicted | observed` gives one string per row, the predicted row on the left of `|` and the observed row on the right. `crop` names that area when it differs from `where`.

When the two grids do not have the same shape, only their shapes are given.

Observations from `data_api` are nested lists. To use a `where` in code, convert the layer first: `numpy.array(obs["frame"][0])[45:50, 34:39]`.

__IMAGE_PREVIEWS__## 4. Follow the next phase

A CWM contradiction (incorrect prediction or reconstruction, or a compactness failure on new evidence) stops the controller call and returns you to **CWM Modeling**. Use the new data to repair and resubmit the CWM. Ordinary exploration completion leaves you in **Active Exploration**: choose the next experiment. Execution failures can require revalidation without demonstrating a wrong game rule; follow the result's `phase_change` field and the reported error. Controller errors receive worst performance metrics; already recorded experience is retained. A successful subgoal does not end the task unless the full game is solved.
