"""Optional local CWM environment template."""

CWM_ENV = r'''"""Local environment backed by your current workspace CWM; no real actions.

EnvCWM(initial_state=state) never queries the dataset.
make_cwm_env() starts where the next controller starts.
Local simulation does not validate or accept code. Submit changed CWM code with
UpdateCodeWorldModel before using RunController.
"""
from copy import deepcopy


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
                "framework.cwm_env.make_cwm_env() to load the recorded start."
            )
        self.obs_mode = obs_mode
        self._initial_state = deepcopy(initial_state)
        self.reset()

    @property
    def state(self):
        return deepcopy(self._state)

    def observation(self):
        """Render the full observation; does not advance the simulation."""
        from world_model import model_render
        return deepcopy(model_render.render(self.state))

    def reset(self, initial_state=None):
        """Restart from the default state or from the supplied State for this episode."""
        self._state = deepcopy(self._initial_state if initial_state is None else initial_state)
        return self.observation() if self.obs_mode else self.state

    def step(self, action):
        """Apply one problem-format action in the CWM only."""
        if self.observation()["is_done"]:
            raise RuntimeError("The simulated environment has ended; call reset() before step().")
        from world_model import model_transition
        self._state = deepcopy(model_transition.step(self.state, deepcopy(action)))
        return self.observation() if self.obs_mode else self.state


def make_cwm_env(obs_mode=False, initial_state=None):
    """Create EnvCWM starting where the next RunController will start, unless a State is given.

    single_instance: the current State, rebuilt with your workspace CWM from the live episode
    (get_initial_state on its first observation, then step through every recorded action).
    multi_instance: get_initial_state(initial observation). This factory is for workspace
    scripts, not submitted callbacks. Further resets never query the dataset.
    """
    if initial_state is None:
        from framework import data_api
        from world_model import model_initial_state, model_transition
        summary = data_api.summary()
        if summary["lifecycle"] == "single_instance":
            observations, actions = data_api.load_history(summary["episode_id"])
        else:
            observations = data_api.load_observations([summary["initial_observation_id"]])
            actions = []
        initial_state = model_initial_state.get_initial_state(deepcopy(observations[0]))
        for action in actions:
            initial_state = model_transition.step(initial_state, deepcopy(action))
    return EnvCWM(obs_mode=obs_mode, initial_state=initial_state)
'''
