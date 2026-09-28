# Workflow

To solve the game, build a Code World Model (CWM): Python code that represents the game's state, reconstructs observations and predicts the effects of actions. Improve it as you explore and make progress.

- **Recorded experience.** Real observations and transitions are automatically recorded in an external dataset accessible through `framework.data_api`. Read `framework/data_api.py` for API documentation and examples.
- **Initial situation.** A random policy already prepared the dataset, targeting __INITIAL_TARGET__ unique observations (possibly fewer if a collection limit was reached). Call `data_api.summary()` to inspect the current dataset. You begin in **CWM Modeling**.
- **CWM Modeling.** Build or update the CWM in `world_model/`, then submit it with `UpdateCodeWorldModel`. It must exactly match all recorded experience and satisfy the other checks described in `docs/CWM_modeling_phase.md`. Acceptance takes you to **Active Exploration**.
- **Active Exploration.** Produce a controller in `exploration.py` aiming to explore new and relevant states. Describe the experiment's goal in its module docstring, then call `SubmitExplorationController`. The controller is simulated in the CWM before real execution from the fixed starting observation. Real contradictions return you to **CWM Modeling**; ordinary exploration completion leaves you in **Active Exploration**. Read `docs/active_exploration_phase.md` for the full process.
- **Optional planning.** In Active Exploration, define a goal in `goal.py` and use `PlanInCWM` to find an action list in the accepted CWM. You can then use that list in your exploration controller. See `docs/plan_in_CWM.md`.
- **Final objective.** Fully complete the game. Keep working until the game is solved or the framework ends the task (for example, when its budget is exhausted). Seek useful understanding as well as immediate progress: test mechanisms and hypotheses rather than collecting arbitrary new observations just for novelty.

Submitted callbacks cannot access the dataset, live environment, workspace files or network. Use recorded data while developing; submitted code must operate on its inputs and imported Python code/constants. Read the docstrings in `framework/data_api.py` for data access, field definitions and examples. The phase documents explain code execution rules and limits.
