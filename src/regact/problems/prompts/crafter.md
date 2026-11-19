# Game: Crafter

Crafter is an open-world survival game on a 2D grid. You control one player who can move, collect resources, craft tools, place objects, eat, drink, sleep and fight.

**Goal:** unlock as many of the {n_achievements} achievements as you can within a life. Each achievement counts once per life. The achievements are: {achievements}.

The game does not tell you how to unlock them: find out by playing. The world is random and creatures move on their own, so the same actions do not always give the same result.

## Observation

`obs.frame` is a dictionary:

- `view`: the map around the player, 7 rows of 9 cells, row 0 at the top (up) and column 0 at the left. The player is always at the centre, `view[3][4]`. Each cell is a name: a terrain (`grass`, `water`, `sand`, `stone`, `path`, `tree`, `lava`, `coal`, `iron`, `diamond`, `table`, `furnace`), or what stands on it (`player`, `cow`, `zombie`, `skeleton`, `arrow-left` (or `-right`, `-up`, `-down`), `plant`, `plant-ripe`), or `void` beyond the edge of the world.
- `facing`: the direction the player faces: `left`, `right`, `up` or `down`.
- `inventory`: counts of everything the player has. `health`, `food`, `drink` and `energy` go from 0 to 9.
- `sleeping`: whether the player is asleep.
- `daylight`: from 0 (night) to 1 (day).

`obs.info["achievements"]` lists the achievements unlocked so far in the current life. `obs.reward` is +1 when an achievement is unlocked for the first time, and it follows health changes (+0.1 or -0.1 per point).

## Actions

{actions}.

A move turns the player to face that direction, and also moves it one cell when that cell is free. `do` acts on the cell the player faces. The `place_` and `make_` actions do nothing when their conditions are not met.

## End of a life

The life ends when `health` reaches 0 (`obs.is_done` is true) or after 10000 actions. A new life starts with an empty inventory and nothing unlocked.
