"""Block 2 manual smoke: drive a FakeNativeEnv through an EnvSession.

Run:  python debug/block2_env.py
"""

from regact.env.lifecycle import MultiInstancePolicy
from regact.env.renderer import RawRenderer
from regact.env.session import EnvSession
from regact.testing.fakes import FakeNativeEnv


def main() -> None:
    session = EnvSession(
        make_native=lambda: FakeNativeEnv(goal=3),
        key="corridor",
        renderer=RawRenderer(),
        lifecycle=MultiInstancePolicy(),
    )
    env = session.make()
    obs = session.reset()
    print("reset:", obs.frame, "| actions:", obs.available_actions)
    step = 0
    while not env.is_done:
        obs = env.step(1)
        step += 1
        print(f"step {step}: grid={obs.frame['grid']} reward={obs.reward} done={obs.is_done}")
    print(f"done in {env.action_count} actions, final reward {obs.reward}")


if __name__ == "__main__":
    main()
