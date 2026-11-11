# Environment

The environment is a **deterministic POMDP starting from a single initial state**. Its rules and its goal are not given; discover them by interacting with it. It may or may not have several levels: completing a level loads the next one, and completing the last level wins the game.

Every observation is a dictionary with five keys:

- `frame`: what the environment shows, a JSON value (nested lists, numbers, strings or dictionaries). Its structure and meaning are part of what you discover.
- `reward`: the reward of the action that led to this observation.
- `is_done`: true when the game has ended.
- `available_actions`: the actions accepted now.
- `info`: other information the environment reports. `info["milestones"]` lists the events produced by the action that led to this observation; it is not cumulative.

__ACTION_FORMAT__

Your score on each level depends on the number of actions you use before completing it: fewer is better. Actions spent exploring count as much as those that complete the level. A reset, where the environment has one, is not counted.
