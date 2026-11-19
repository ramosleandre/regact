# Environment

The environment is a **stochastic, partially observable open world**: it changes on its own, so the same actions do not always lead to the same observation. Its rules are not given; discover them by interacting with it. It has no levels.

Every observation is a dictionary with five keys:

- `frame`: what the environment shows, a JSON value (nested lists, numbers, strings or dictionaries). Its structure and meaning are part of what you discover.
- `reward`: the reward of the action that led to this observation.
- `is_done`: true when the current life has ended.
- `available_actions`: the actions accepted now.
- `info`: other information the environment reports. `info["milestones"]` lists the events produced by the action that led to this observation; it is not cumulative. `info["achievements"]` lists the achievements unlocked so far in the task, and `info["life_achievements"]` those unlocked in the current life.

An action is an integer from `available_actions`.

{score} Which achievements exist and how to unlock them is for you to discover; each new one is reported as a milestone. A life ends when the player dies or after 10000 actions. A reset then starts a new life {reset}: the inventory is lost, the achievements already unlocked are kept. {won}
{evaluation}
