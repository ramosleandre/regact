"""Optional local CWM environment and editable simulation script templates."""

CWM_ENV = r'''"""Local environment backed by your current workspace CWM; no real actions.

EnvCWM(initial_state=state) never queries the dataset.
make_cwm_env() starts where the next controller starts (replaying the live chain in
single_instance). replay_episode(episode_id) shows where your CWM first diverges from a
recorded episode.
For an editable controller loop with printed states/actions, see simulate.py.
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

    single_instance: the current State, rebuilt by replaying the live chain through your
    workspace CWM (parse its first observation, then step through every recorded action).
    multi_instance: parse(initial observation). This factory is for workspace scripts, not
    submitted callbacks. Further resets never query the dataset.
    """
    if initial_state is None:
        from framework import data_api
        summary = data_api.summary()
        if summary["lifecycle"] == "single_instance":
            initial_state = replay_episode(summary["episode_id"], quiet=True)["state"]
        else:
            from world_model import model_parser
            obs = data_api.load_observations([summary["initial_observation_id"]])[0]
            initial_state = model_parser.parse(obs)
    return EnvCWM(obs_mode=obs_mode, initial_state=initial_state)


def replay_episode(episode_id, quiet=False):
    """Replay a recorded episode through your WORKSPACE CWM, as UpdateCodeWorldModel does, and
    report the first divergence. Replays the whole chain up to that episode: parse its first
    observation, step through every action, and at an explicit reset call your optional
    reset(state, kind) hook (or parse the reset observation). Returns a dict: diverged (bool),
    episode_id, step (actions applied in that episode; 0 = its first observation), state (the
    State there; on divergence, the State that rendered wrong) and, on divergence, state_before,
    predicted, observed and differences (up to 20). Prints a short report unless quiet (the
    printed State is cut at 1,000 characters; the returned one is complete)."""
    from framework import data_api
    from world_model import model_parser, model_render, model_transition
    episodes = data_api.list_episodes()
    target = next(e for e in episodes if e["episode_id"] == episode_id)
    chain = [e for e in episodes if e["chain_id"] == target["chain_id"] and e["episode_id"] <= episode_id]
    reset = getattr(model_transition, "reset", None)
    state = None
    for segment in chain:
        observations, actions = data_api.load_history(segment["episode_id"])
        before = state
        if segment["continues_episode"] is not None and callable(reset) and state is not None:
            state = reset(state, segment["started_by"].removeprefix("reset_"))
        else:
            state = model_parser.parse(deepcopy(observations[0]))
        result = _compare(segment["episode_id"], 0, before, state, observations[0], quiet)
        if result:
            return result
        for t, action in enumerate(actions, start=1):
            before = state
            state = model_transition.step(state, deepcopy(action))
            result = _compare(segment["episode_id"], t, before, state, observations[t], quiet)
            if result:
                return result
    if not quiet:
        print(f"Episode {episode_id}: your CWM reproduces every recorded observation.")
    return {"diverged": False, "episode_id": episode_id, "step": len(actions), "state": state}


def _compare(episode_id, step, before, state, observed, quiet):
    from world_model import model_render
    predicted = model_render.render(state)
    if predicted == observed:
        return None
    diffs = []
    def visit(a, b, path):
        if len(diffs) >= 20:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b), key=str):
                visit(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}")
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{i}]")
        elif a != b:
            diffs.append({"path": path.lstrip("."), "predicted": a, "observed": b})
    visit(predicted, observed, "")
    if not quiet:
        print(f"Episode {episode_id}, step {step}: first divergence ({len(diffs)} differences shown).")
        text = repr(before)
        print(f"State before: {text if len(text) <= 1000 else text[:1000] + ' ...'}")
        for d in diffs:
            print(f"  {d['path']}: predicted {d['predicted']!r}, observed {d['observed']!r}")
    return {
        "diverged": True, "episode_id": episode_id, "step": step, "state": state,
        "state_before": before, "predicted": predicted, "observed": observed, "differences": diffs,
    }
'''

SIMULATE = r'''"""Edit this script to inspect your controller in the current workspace CWM.

Run: python simulate.py --max-actions 20 (or --max-actions null for no action cap).
Customize prints, the controller or its starting state. This script takes no
real actions, records no experience and does not validate/accept the CWM.
The shell tool's timeout applies, not the submitted-callback/controller-call budgets.
Restart the script after editing CWM files; imports are not reloaded in place.
"""
from framework.cwm_env import make_cwm_env

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
    import controller
    run_controller(controller.get_controller(), make_cwm_env(), args.max_actions)
'''
