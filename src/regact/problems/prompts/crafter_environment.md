# Environment

The environment is the open-world survival game **Crafter**. It is stochastic and partially observable: it changes on its own, so the same actions do not always lead to the same observation. It has no levels.

Every observation is a dictionary with five keys:

- `frame`: a dictionary. `view` is the map window around the player, 7 rows of 9 names (the terrain or the creature on each cell, `player` at the centre); `facing` is the direction the player faces (`left`, `right`, `up` or `down`); `inventory` counts what the player has, including `health`, `food`, `drink` and `energy`; `sleeping` and `daylight` complete it.
- `reward`: the reward of the action that led to this observation.
- `is_done`: true when the current life has ended.
- `available_actions`: the actions accepted now.
- `info`: other information the environment reports. `info["milestones"]` lists the events produced by the action that led to this observation; it is not cumulative. `info["achievements"]` lists the achievements unlocked so far in the current life.

An action is one of these names, as a string:

{actions}

These are the achievements of the game: {achievements}.

{score} A life ends when the player dies or after {life} actions. A reset then starts a new life {reset}, with an empty inventory and nothing unlocked.
{evaluation}
