# Reading differences

When a prediction does not match reality, the result's `message` says in one sentence where it happened and what differs. The counterexample gives the details, and `differences` lists what differs between the predicted observation and the observed one.

## Where it happened

- `episode_id`, `step`: the episode and the number of actions applied in it.
- `from_observation_id`, `action`, `observation_id`, `transition_id`: the step itself; the State before it was carried from `from_observation_id`, and `action` was applied.
- `explained_by_previous_version` (refused `UpdateCodeWorldModel` only): `true` when your previously accepted CWM predicted this step correctly, so the latest change broke it; `false` when no accepted version has explained it yet.
- `diagnostic_id` and `details`: `details` is the call that describes this diagnostic: the same differences with other limits if you pass them, and your model's State before the step and the State it predicted. `data_api.load_diagnostic(diagnostic_id)` returns the complete record, including both full observations; it can be very large.__DIAGNOSTIC_IMAGE__

## What differs

`differences` lists every field of the observation that differs (`frame`, `info`, `is_done`, ...). Each field is reported separately and within its own limits, so a large frame difference never hides a difference in another field.

Values are listed by path, for example `{"where": "info.state", "predicted": "NOT_FINISHED", "observed": "GAME_OVER"}`. At most __DIFF_ITEMS__ values are listed per field; `other differences` counts the rest.

A grid, such as one frame layer, is summarized as regions. A region is a group of differing cells that touch, including diagonally. `cells_differing` counts the differing cells of the grid and `cells_total` all its cells. At most __DIFF_REGIONS__ regions are listed, largest first; `other_regions` counts the rest.

Each region gives:

- `where`, in Python slice notation with exclusive ends, row first, then column (on a screen: y, then x):
  - `frame[0][45:50, 34:39]`: the rectangle of rows 45 to 49 and columns 34 to 38 of layer 0;
  - `frame[0][63, 0:64]`: one row; `frame[0][63, 63]`: one cell;
  - `frame[0][0:4, 0] + frame[0][3, 1:4]`: exactly these cells, as a sum of rectangles;
  - `within frame[0][9:45, 27:57]`: an irregular region somewhere inside this rectangle, not filling it.
- `cells`: how many cells of the region differ.
- Either `changes`, the `predicted -> observed` value pairs with their cell counts (at most __DIFF_CHANGES__, then `other changes`), or, for a small region holding several changes or an irregular shape, the area itself: `predicted | observed` gives one string per row, the predicted row on the left of `|` and the observed row on the right. `crop` names that area when it differs from `where`.

When the two grids do not have the same shape, only their shapes are given.

Observations from `data_api` are nested lists. To use a `where` in code, convert the layer first: `numpy.array(obs["frame"][0])[45:50, 34:39]`.
