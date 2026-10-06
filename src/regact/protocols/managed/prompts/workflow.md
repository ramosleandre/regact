# Workflow

__PROTOCOL_INTRO__- **Recorded experience.** Real observations and transitions are automatically recorded in an external dataset accessible through `framework.data_api`. Read `framework/data_api.py` for API documentation, field definitions and examples.
- **Initial situation.** A random policy already prepared the dataset, targeting __INITIAL_TARGET__ unique observations (possibly fewer if a collection limit was reached). Call `data_api.summary()` to inspect the current dataset. __INITIAL_GUIDANCE__
__PROTOCOL_STEPS__
- **Final objective.** Fully complete the game. Keep working until the game is solved or the framework ends the task (for example, when its budget is exhausted). Seek useful understanding as well as immediate progress: test mechanisms and hypotheses.

__LIFECYCLE__

Submitted callbacks cannot access the dataset, live environment, workspace files or network. Use recorded data while developing; submitted code must operate on its inputs and imported Python code/constants. The controller may retain internal memory during one call.

Feedback includes recorded observation and transition IDs, performance metrics, new milestones and saved image paths when available. Errors preserve already recorded experience; inspect the reported error and evidence before retrying.
