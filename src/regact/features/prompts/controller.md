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

Implement your controller in `solution.py`, then run **SubmitSolution** to score it
against the environment; you may submit as many times as you like. Call **ExitTask** to
finish the run once you are satisfied. See *Framework tools* below for how to run these.

**Be a policy, not a search.** Your controller is judged as a *policy* `act(obs) -> action`
in the reinforcement-learning sense: it must *react to the observation* and generalise. Do
**not** run a search or exploration algorithm (BFS/DFS, exhaustive sweeps) over the
environment to discover a fixed winning sequence and then replay it — a hard-coded,
brute-forced path ignores `obs` and is exactly the failure mode that scores poorly. Be
**economical with environment interactions**: probe with curiosity only enough to infer the
dynamics, then encode that rule as the policy so it works from any state.
