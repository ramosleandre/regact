## Your deliverable: a controller

This task gives you one feature — a **controller**: a pure policy `act(obs) -> action`.
It is re-instantiated for each episode (so `__init__` is your per-episode reset; keep
state on `self`), and it never receives or imports the environment — it acts only on the
`obs` it is given.

Your working directory already holds the files to edit:

- `solution.py` — the stub you implement and submit (a `Controller` class and a
  `get_controller()` factory).
- `code_library/base_controller.py` — the contract to subclass.
- `code_library/example_controller.py` — a trivial worked example.

Implement your controller in `solution.py`, then run **SubmitSolution** to score it
against the environment; you may submit as many times as you like. Call **ExitTask** to
finish the run once you are satisfied. See *Framework tools* below for how to run these.
