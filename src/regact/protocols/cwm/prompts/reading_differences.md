# Reading differences

When a prediction does not match reality, the counterexample says where it happened and `differences` says what differs between the predicted observation and the observed one.

## Where it happened

- `episode_id`, `step`: the episode and the number of actions applied in it.
- `from_observation_id`, `action`, `observation_id`, `transition_id`: the step itself; the State before it was carried from `from_observation_id`, and `action` was applied.
- `explained_by_previous_version` (refused `UpdateCodeWorldModel` only): `true` when your previously accepted CWM predicted this step correctly, so the latest change broke it; `false` when no accepted version has explained it yet.
- `full_data`: the call that returns the complete predicted and observed observations.

## What differs

`differences` lists every field of the observation that differs (`frame`, `info`, `is_done`, ...). Each field is reported separately and within its own limits, so a large frame difference never hides a difference in another field.

Values are listed by path, for example `{"where": "info.state", "predicted": "NOT_FINISHED", "observed": "GAME_OVER"}`. At most __DIFF_ITEMS__ values are listed per field; `other differences` counts the rest.

A grid, such as one frame layer, is summarized as regions. A region is a group of differing cells that touch, including diagonally. `cells_differing` counts all differing cells of the grid. At most __DIFF_REGIONS__ regions are listed, largest first; `other_regions` counts the rest.

Each region gives:

- `where`, in Python slice notation with exclusive ends:
  - `frame[0][45:50, 34:39]`: the rectangle of rows 45 to 49 and columns 34 to 38 of layer 0;
  - `frame[0][63, 0:64]`: one row; `frame[0][63, 63]`: one cell;
  - `frame[0][0:4, 0] + frame[0][3, 1:4]`: exactly these cells, as a sum of rectangles;
  - `within frame[0][9:45, 27:57]`: an irregular region somewhere inside this rectangle, not filling it.
- `cells`: how many cells of the region differ.
- Either `changes`, the `predicted -> observed` value pairs with their cell counts (at most __DIFF_CHANGES__, then `other changes`), or, for a small region holding several changes or an irregular shape, the area itself: `predicted` and `observed` give one string per row, values separated by spaces. `crop` names that area when it differs from `where`.

When the two grids do not have the same shape, only their shapes are given.
