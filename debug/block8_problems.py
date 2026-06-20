"""Block 8 manual smoke: build problems, show prompt + renderer, drive MiniGrid if present.

Run:  python debug/block8_problems.py
"""

from regact.config.schema import InfoMode, ObsMode
from regact.problems.base import build_problem
from regact.problems.minigrid import MiniGridRenderer


def main() -> None:
    problem = build_problem("minigrid", {"env_id": "MiniGrid-Empty-5x5-v0"})
    print("problem:", problem.name, "| tasks:", problem.get_task_names())
    print("config_kwargs:", problem.config_kwargs())
    print("renderer:", type(problem.obs_renderer("t", mode=ObsMode.RAW)).__name__)

    print("\n--- build_prompt (informative) ---")
    print(problem.build_prompt("MiniGrid-Empty-5x5-v0", info_mode=InfoMode.INFORMATIVE))
    print("\n--- build_prompt (minimal) ---")
    print(problem.build_prompt("MiniGrid-Empty-5x5-v0", info_mode=InfoMode.MINIMAL))

    # Renderer is JSON-safe even without the lib (numpy-like obs -> nested lists).
    obs = MiniGridRenderer().render(
        {"image": [[0, 1], [1, 0]], "direction": 2}, {"available_actions": [0, 1, 2]}
    )
    print("\nrendered obs frame:", obs.frame, "| actions:", obs.available_actions)

    try:
        import minigrid  # noqa: F401

        native = problem.make_env("MiniGrid-Empty-5x5-v0")
        first, info = native.reset(seed=0)
        print("\nmake_env OK — available_actions:", info["available_actions"])
        native.close()
    except ImportError:
        print("\n(minigrid not installed — skipping live make_env)")

    print("\nARC is deferred to Block 8b:")
    try:
        build_problem("arc_agi", {})
    except NotImplementedError as exc:
        print("  build_problem('arc_agi') ->", exc)


if __name__ == "__main__":
    main()
