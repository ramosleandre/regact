# Environment interface

You reach the environment only through `framework/make_env.py`. `make_env()` returns an environment client; `env.current()`, `env.reset()`, and `env.step(action)` each return exactly ONE `Obs` object - never a Gym-style tuple, so do not unpack their return value. An `Obs` has:
- `obs.frame` - the game's native observation (its structure is game-specific; see the game section)
- `obs.reward` - reward from the preceding action (may be `None`)
- `obs.is_done` - whether the episode has ended
- `obs.available_actions` - action IDs accepted by this task (an accepted action may still have no effect when its preconditions are not met)
- `obs.info` - extra metadata

```python
env = make_env()
obs = env.current()  # the current state, without spending an action
obs = env.step(obs.available_actions[0])
print(obs.reward, obs.is_done)
```
