# Active Exploration

**Goal:** explore new and relevant states, test uncertain game mechanisms and make progress toward solving the game. Choose a concrete experiment. New observations are required for real execution, but arbitrary novelty is not an objective in itself.

## 1. Write `exploration.py`

Put a short, plain-text description of the experiment in the file's **module docstring** (the first triple-quoted string). Define `get_controller()` returning a new object with:

- `act(state)`: return one action in the game's format, using a CWM State as input.
- `is_done(state)`: return True to stop this exploration before its next action. The default False is valid: the game ending or an action/time limit can stop it. Add an earlier stopping condition when it suits your experiment.

The controller may retain internal memory. It receives a fresh instance for simulation and another fresh instance for real execution. Real observations are parsed into States by the accepted CWM before being passed to your controller. Controllers and goals cannot query the dataset or the live environment themselves.

Optional `objective_reached(state) -> bool` reports whether your particular goal was achieved. It is independent of `is_done`: stopping because a list ran out does not establish success. You do not need this optional method to submit a controller.

For a controller built from a planned action list, see `docs/plan_in_CWM.md`.

## 2. Submit the controller

Run `python framework/control.py SubmitExplorationController` with no arguments. The command uses the accepted, frozen CWM, even if you have edited `world_model/`. It performs these two stages:

1. **Simulation:** start from the recorded fixed initial observation and run your controller inside the CWM. At least one predicted complete observation must be outside the current dataset. If simulation fails or predicts no novelty, the command stops here; no real actions are taken.
2. **Real execution:** start a fresh real episode from that same starting observation, with a fresh controller. Record every transition and check it against the CWM. Stop at the first contradiction, controller completion, game end or limit.

## 3. Read the two-stage feedback

The result first reports whether simulation succeeded and how many new observations it predicted. On failure, a structured explanation follows; repair the code or choose a more informative experiment. On success, the real-execution result follows:

- `observation_ids` and `transition_ids`: all recorded evidence encountered in this real episode, including reused IDs and its starting observation. Load these directly with `data_api`; `[4:9]` means IDs 4 through 9, inclusive.
- Counts distinguish simulated actions/novelty from actual actions/novelty. Predictions never enter the real dataset.
- `metrics`: actual game performance. New-milestone notices highlight first-time progress/failure events in real experience. Simulated achievements do not count.
- `goal_achieved`: your controller's optional goal check, not necessarily game completion. Finishing an action list also does not imply its goal was achieved.
- Counterexample/diagnostic IDs: evidence to investigate when a check or callback fails.

Up to __IMAGE_COUNT__ PNG previews are saved in `tmp/images/obs_id_<ID>.png`, selected from the first and last distinct observations encountered. `observation_images` lists the saved paths. Read them with your image tool. This folder is emptied at every new submission, including a refused one; copy images elsewhere if needed. The dataset itself remains available regardless of preview cleanup.

## 4. Follow the next phase

A CWM contradiction (incorrect prediction/reconstruction, state collision or a compactness failure on new evidence) stops the episode and returns you to **CWM Modeling**. Use the new data to repair and resubmit the CWM. Ordinary exploration completion leaves you in **Active Exploration**: choose the next experiment. Execution failures can require revalidation without demonstrating a wrong game rule; follow the separate phase notice and the reported error. Controller errors receive worst performance metrics; already recorded experience is retained. A successful subgoal does not end the task unless the full game is solved.
