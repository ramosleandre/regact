## The grid: egocentric symbolic view (not RGB pixels)

`obs.frame["image"]` is a nested Python list (it crossed the JSON boundary, so it is NOT a
NumPy array and has no `.shape`), and it is a SYMBOLIC encoding, not pixels. Its logical
shape is `[7][7][3]` - the agent's EGOCENTRIC view, not the whole map. The agent sits at
`image[3][6]` and looks toward DECREASING second index: `image[3][5]` is one cell ahead,
`image[3][0]` the furthest visible cell ahead, `image[4][6]` one cell to its right,
`image[2][6]` one to its left. The view ROTATES with the agent, so "ahead" is always the
smaller second index. The agent's own cell is NOT drawn as object 10 here (it shows as
empty, id 1); use `obs.frame["direction"]` for the heading. Cells hidden behind walls are
occluded (object `0 unseen`). A carried object appears at the agent's own view position.

Each cell is `[object, color, state]`:
- object: `0 unseen, 1 empty, 2 wall, 3 floor, 4 door, 5 key, 6 ball, 7 box, 8 goal, 9 lava, 10 agent`
- color:  `0 red, 1 green, 2 blue, 3 purple, 4 yellow, 5 grey`
- state (doors only): `0 open, 1 closed, 2 locked`

`obs.frame["direction"]` is the agent's world heading: `0 right (+x), 1 down (+y),
2 left (-x), 3 up (-y)`.
