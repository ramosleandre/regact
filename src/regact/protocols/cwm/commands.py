"""Shared command descriptions for native tools and generated terminal help."""

from regact.protocols.cwm.config import CwmConfig

COMMANDS = {
    "UpdateCodeWorldModel": "Validate world_model/ against all recorded real experience.",
    "PlanInCWM": "Find an action list in the accepted CWM using goal.py; no real actions.",
    "RunController": (
        "Run a fresh controller from controller.py, checking real steps against the accepted CWM."
    ),
}


def enabled_commands(options: CwmConfig) -> dict[str, str]:
    """Resolve the tools taught and exposed by this run's CWM configuration.

    COMMANDS remains the complete catalogue so saved runs stay recognizable by
    the viewer, independently of the defaults used for new experiments.
    """
    return {
        name: description
        for name, description in COMMANDS.items()
        if name != "PlanInCWM" or options.planner.enabled
    }
