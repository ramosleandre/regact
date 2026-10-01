# Optional planning in the CWM

Use this during **Active Exploration** when you can express an experiment as a goal on States. Planning uses the accepted CWM and takes no real actions. You can always write a controller directly instead.

## 1. Define the fixed file `goal.py`

Describe your experimental goal in its **module docstring**. Implement:

```python
def achieved(state):
    # Return whether this state meets your chosen objective.
    ...

# Optional:
def utility(state):
    # Return a number in [0, 1]; greater means a more desirable endpoint.
    ...
```

Both functions must be repeatable and must not mutate the input. If utility is omitted, it defaults to `float(achieved(state))`. An achieved goal must have utility 1; utility 1 does not necessarily mean the goal is achieved. Use a graded utility when partial progress can help guide the choice of an incomplete candidate.

## 2. Run `PlanInCWM`

Run `python framework/commands.py PlanInCWM` with **no arguments**. It reads `goal.py` and the accepted CWM; edits to `world_model/` take effect only after another successful `UpdateCodeWorldModel`.

The planner starts from the observation where the next RunController would start: the current live observation in single-instance mode, or the original initial observation in multi-instance mode. It reads that recorded observation without taking a real action. It searches for an action list that visits at least one predicted observation outside the current dataset. It maximizes the final state's utility, then minimizes the number of actions. Utility is not summed along the path.

The search algorithm is **__PLANNER_ALGORITHM__** (breadth-first search). It enumerates the problem's available actions, including valid coordinate actions when present; large action spaces can exhaust the budget quickly. Graded utility ranks candidates; it does not turn breadth-first search into a heuristic-guided algorithm.

Configured search limits:

| Limit | Value |
|---|---|
| Whole planner call, including startup and goal evaluation | __PLANNER_SECONDS__ seconds |
| CWM `parse`/`step`/`render` calls | __PLANNER_CALLS__ |
| Stored search states (including the starting state) | __PLANNER_NODES__ |
| Maximum action-list length | __PLANNER_DEPTH__ |

Goal evaluations use time but do not count against the CWM-call cap. Per-callback and memory limits also apply (see the code execution rules in `docs/CWM_modeling_phase.md`). An unlimited individual cap does not disable the other limits.

## 3. Inspect the result, then use the plan

When a candidate exists, the result gives a path such as `plans/plan_003.py`, which contains `ACTIONS`. Check these fields independently:

- `goal_achieved`: whether the candidate's endpoint met your `achieved` function.
- `optimality_proven`: whether its utility/action count was proven optimal in the searched CWM/action space. This does not establish correctness in the real game.
- `n_states_searched`: stored search states, including the start.
- `elapsed_seconds`: search time; the whole-call deadline includes startup too.

A budget-limited candidate may still be useful even if it does not reach the goal or optimality is unproven. Read the warnings. No candidate means none was found within the reported search bounds, not that the goal is impossible in the real game. Code errors include evidence for debugging the simulated state/action.

To use the returned file, write `controller.py`:

```python
"""Describe what this action sequence should test or achieve."""
from plans.plan_003 import ACTIONS  # use the path actually returned
from framework.action_list_controller import ExplorationControllerFromListActions
from goal import achieved          # optional; stops early when the goal is reached

def get_controller():
    return ExplorationControllerFromListActions(ACTIONS, achieved=achieved)
```

Then call `RunController`. The framework runs it from the start defined by the task lifecycle with per-action CWM checks; there is no additional simulated episode. Revalidate any changed CWM source before submitting. Omit `achieved=achieved` if you want to execute the entire list; list exhaustion stops it but does not report that a goal was reached. Plans are not automatically submitted and stay predictions from the CWM version used to generate them. Reconsider old plans after accepting a new CWM.
