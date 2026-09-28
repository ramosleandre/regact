"""Optional local simulation template; no trusted framework execution happens here."""

SIMULATION = r'''"""Try the current workspace CWM/controller locally; no real actions are performed.

python framework/simulation.py --max-actions 20

EnvCWM(initial_state=state): pure state/observation simulator with no data access.
make_cwm_env(): convenience factory loading the recorded starting observation once.
run_controller(controller, env): print a local controller's simulated actions.
Read each function/class docstring for arguments and examples.

This development helper does not validate or accept a CWM. PlanInCWM and
SubmitExplorationController still use the last accepted CWM. Local execution
has the agent shell tool's timeout; isolated framework callback limits do not
apply here. Stop a hanging script through that tool and repair the callback.
"""
from copy import deepcopy
from pathlib import Path
import sys

_workdir = Path(__file__).resolve().parents[1]
for _directory in (_workdir, _workdir / "world_model"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))


class EnvCWM:
    """reset()/step(action) return State, or an observation dict if obs_mode=True.

    Unlike Gym, step returns just that value, not a reward/termination tuple.
    Access observation() for reward, is_done and available_actions in either mode.
    reset() restores a copy of the constructor's starting state. reset(state)
    starts this episode elsewhere without changing the default starting state.
    The state property and returned values are copies, so callers cannot mutate
    the environment accidentally. Submitted callbacks must pass initial_state;
    this class never calls the experience API.
    """
    def __init__(self, obs_mode=False, initial_state=None):
        if type(obs_mode) is not bool:
            raise TypeError("obs_mode must be bool")
        if initial_state is None:
            raise ValueError(
                "EnvCWM needs initial_state. In workspace scripts, use "
                "framework.simulation.make_cwm_env() to load the recorded start."
            )
        self.obs_mode = obs_mode
        self._initial_state = deepcopy(initial_state)
        self.reset()

    @property
    def state(self):
        return deepcopy(self._state)

    def observation(self):
        """Render the full observation; does not advance the simulation."""
        import model_render
        return deepcopy(model_render.render(self.state))

    def reset(self, initial_state=None):
        """Restart from the default state or from the supplied State for this episode."""
        self._state = deepcopy(self._initial_state if initial_state is None else initial_state)
        return self.observation() if self.obs_mode else self.state

    def step(self, action):
        """Apply one problem-format action in the CWM only."""
        if self.observation()["is_done"]:
            raise RuntimeError("The simulated environment has ended; call reset() before step().")
        import model_transition
        self._state = deepcopy(model_transition.step(self.state, deepcopy(action)))
        return self.observation() if self.obs_mode else self.state


def make_cwm_env(obs_mode=False, initial_state=None):
    """Create EnvCWM. With no State, read/parse the fixed start once via data_api.

    This factory is for workspace scripts, not submitted callbacks. Passing a
    State skips data access entirely. Further resets never query the dataset.
    """
    if initial_state is None:
        from framework.data_api import summary, load_observations
        import model_parser
        oid = summary()["initial_observation_id"]
        initial_state = model_parser.parse(load_observations([oid])[0])
    return EnvCWM(obs_mode=obs_mode, initial_state=initial_state)


def run_controller(controller, env, max_actions=20):
    """Reset an env-like instance, print every value/action, return a short summary.

    controller.act(value) and optional is_done(value) receive reset/step values:
    use state mode for an ExplorationController. env.observation() must return
    a dict containing is_done. No extra controller methods are required.
    This helper does not reload Python imports: start a new script after edits.
    """
    if max_actions is not None and (type(max_actions) is not int or max_actions < 1):
        raise ValueError("max_actions must be a positive integer or None")
    value = env.reset()
    print(f"Initial: {value!r}", flush=True)
    stop = getattr(controller, "is_done", lambda value: False)
    actions = 0
    while True:
        if env.observation()["is_done"]:
            reason = "environment_done"
            break
        if stop(value):
            reason = "controller_done"
            break
        if max_actions is not None and actions >= max_actions:
            reason = "max_actions"
            break
        action = controller.act(value)
        value = env.step(action)
        actions += 1
        print(f"Action {actions}: {action!r}\nResult: {value!r}", flush=True)
    result = {"actions": actions, "stop_reason": reason}
    print(result, flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-actions", type=lambda value: None if value.lower() == "null" else int(value), default=20)
    args = parser.parse_args()
    import exploration
    run_controller(exploration.get_controller(), make_cwm_env(), args.max_actions)
'''
