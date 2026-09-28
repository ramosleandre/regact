"""Shared command descriptions for native tools and generated terminal help."""

COMMANDS = {
    "UpdateCodeWorldModel": "Validate world_model/ against all recorded real experience.",
    "PlanInCWM": "Find an action list in the accepted CWM using goal.py; no real actions.",
    "SubmitExplorationController": (
        "Check exploration.py in the CWM, then run it in the real environment if accepted."
    ),
}
