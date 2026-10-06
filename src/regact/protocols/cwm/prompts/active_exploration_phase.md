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
- Counterexample/diagnostic IDs: evidence to investigate when a check or callback fails.

__IMAGE_PREVIEWS__## 4. Follow the next phase

A CWM contradiction (incorrect prediction or reconstruction, or a compactness failure on new evidence) stops the controller call and returns you to **CWM Modeling**. Use the new data to repair and resubmit the CWM. Ordinary exploration completion leaves you in **Active Exploration**: choose the next experiment. Execution failures can require revalidation without demonstrating a wrong game rule; follow the result's `phase_change` field and the reported error. Controller errors receive worst performance metrics; already recorded experience is retained. A successful subgoal does not end the task unless the full game is solved.
