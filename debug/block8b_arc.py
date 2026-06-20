"""Block 8b manual smoke: ARC problem (offline, local environnement/).

Shows the discovered games, the prompt (both info modes), the helper, and — if the
arc library is installed — actually makes a game and steps it.

Run:  python debug/block8b_arc.py
"""

from pathlib import Path

from regact.config.schema import InfoMode, ObsMode
from regact.problems.arc_agi.problem import ArcAgiProblem

_ENV_DIR = str(Path(__file__).resolve().parents[1] / "environnement")


def main() -> None:
    problem = ArcAgiProblem(environments_dir=_ENV_DIR)
    games = problem.get_task_names()
    print(f"discovered {len(games)} games:", games)
    print("renderer:", type(problem.obs_renderer("ls20", mode=ObsMode.RAW)).__name__)

    print("\n--- build_prompt(ls20, informative) ---")
    print(problem.build_prompt("ls20", info_mode=InfoMode.INFORMATIVE))
    print("\n--- build_prompt(ls20, minimal) ---")
    print(problem.build_prompt("ls20", info_mode=InfoMode.MINIMAL))

    print("\n--- helper_templates ---")
    for tmpl in problem.helper_templates("ls20"):
        print(f"# {tmpl.relpath}\n{tmpl.content}")

    try:
        import arc_agi  # noqa: F401

        print("\n--- live make_env(ls20) ---")
        native = problem.make_env("ls20")
        obs, info = native.reset()
        print("reset info:", info)
        actions = info.get("available_actions", [])
        if actions:
            _o, reward, terminated, truncated, info2 = native.step(actions[0])
            print(
                f"step({actions[0]}) -> reward={reward} done={terminated} state={info2.get('state')}"
            )
        native.close()
    except ImportError:
        print("\n(arc_agi not installed — skipping live make_env; pip install arc-agi to run it)")


if __name__ == "__main__":
    main()
