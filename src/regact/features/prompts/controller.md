### Controller

A **controller** is a pure policy `act(obs) -> action`. You do not play the game by hand:
you **write the controller code** that chooses the actions, then run and submit it. Do not
reply with actions yourself.

It is re-instantiated for each episode (so `__init__` is your per-episode reset; keep state
on `self`), and it never receives or imports the environment — it acts only on the `obs` it
is given.

Your working directory already holds the files to edit:

- `solution.py` — the stub you implement and submit (a `Controller` class and a
  `get_controller()` factory).
- `code_library/base_controller.py` — the contract to subclass.
- `code_library/example_controller.py` — a trivial worked example.

Implement your controller in `solution.py` first: `act` must return a real action (one of
`obs.available_actions`) — **never submit the unimplemented stub** (a controller that
raises scores nothing). Then run **SubmitSolution** to score it. Once `act` works, **submit
early and often**: submit as soon as it does something sensible, and again after each
improvement — do not refine for a long time without submitting. Only submitted controllers
are scored, and each submission is recorded (score + a video), so frequent submits keep a
trace of progress. Call **ExitTask** to finish the run once you are satisfied. See
*Framework tools* below for how to run these.
