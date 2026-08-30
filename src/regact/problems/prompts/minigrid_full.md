## The grid: symbolic cell encoding (not RGB pixels)

`obs.frame["image"]` is a nested Python list (it crossed the JSON boundary, so it is NOT a NumPy array and has no `.shape`), and it is a SYMBOLIC encoding, not pixels. Its logical shape is `[W][H][3]`: access world position `(x, y)` as `image[x][y]`, where `x` increases to the RIGHT and `y` increases DOWNWARD. The map is fixed in world coordinates and does NOT rotate when the agent turns.

Each cell is `[object, color, state]`:
- object: `0 unseen, 1 empty, 2 wall, 3 floor, 4 door, 5 key, 6 ball, 7 box, 8 goal, 9 lava, 10 agent`
- color:  `0 red, 1 green, 2 blue, 3 purple, 4 yellow, 5 grey`
- state (doors only): `0 open, 1 closed, 2 locked`

The agent's own cell has object id `10`; the STATE channel of that cell holds its facing direction, which is also `obs.frame["direction"]`: `0 right (+x), 1 down (+y), 2 left (-x), 3 up (-y)`. In this fully observable mode the whole map is visible and the carried object is not shown in the grid (track it yourself).
