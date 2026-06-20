"""Block 1 manual smoke: build the core types and print their JSON.

Run:  python debug/block1_types.py
"""

from regact.config.schema import AgentConfig, AgentName, ProblemConfig, RunConfig
from regact.controllers.summary import ControllerRun, MilestoneEvent, ControllerSummary
from regact.envclient.obs import Obs
from regact.obs.result import EpisodeResult, EvalResult
from regact.obs.errors import ErrorCategory, LogComponent, LogRecord


def main() -> None:
    obs = Obs(frame=[[0, 1], [1, 0]], reward=1.0, available_actions=[0, 1, 2])
    print("Obs:", obs.to_json())
    print("Obs roundtrip ok:", Obs.from_json(obs.to_json()) == obs)

    rec = LogRecord(
        ts="2026-01-01T00:00:00", component=LogComponent.ENV_SERVER, level="INFO", event="reset"
    )
    print("LogRecord:", rec.to_json())

    summary = ControllerSummary(
        stop_kind="env_done",
        stop_reason="solved",
        total_steps=12,
        history=ControllerRun(
            name="Nav", events=[MilestoneEvent(step=8, description="levels 0->1")]
        ),
        final_obs=obs,
    )
    print(
        "ControllerSummary:",
        repr(summary),
        "| milestones:",
        [m.description for m in summary.milestones],
    )

    res = EvalResult(
        task="ls20",
        aggregate={"n_episodes": 1, "success_rate": 1.0},
        episodes=[EpisodeResult(episode=0, metrics={"success": True, "levels_completed": 1})],
        executor="in_process",
    )
    print("EvalResult:", res.to_json())

    cfg = RunConfig(
        agent=AgentConfig(name=AgentName.SCRIPTED), problem=ProblemConfig(name="arc_agi")
    )
    print("RunConfig features:", cfg.features)
    print("ErrorCategory values:", [e.value for e in ErrorCategory])


if __name__ == "__main__":
    main()
