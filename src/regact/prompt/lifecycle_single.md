## This run: one persistent environment

There is a single environment for the whole run: `make_env()` returns the same handle
(it is created once). It persists across resets, and the action count **carries across
resets** — wasted actions are permanent. A reset returns the environment toward its
start without creating a new one. So decide deliberately: continue from where you are,
or reset. Progress you have already made is not discarded automatically — it is one
continuous attempt, not a set of independent tries.

If your game has levels, a reset returns you to the current level (completed levels stay
completed), so you can build your policy up level by level.
