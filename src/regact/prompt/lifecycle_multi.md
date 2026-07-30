## This run: fresh episodes

Each `make_env()` and each `env.reset()` starts a **fresh, independent episode** on a new
environment instance. Your controller is re-created for every episode and scored as the
mean over many resets, so it must **generalize**: read the observation instead of
hard-coding a fixed action sequence or memorizing a single layout (whether layouts vary
between resets depends on the game — do not assume either).

If your game has levels, every episode restarts from the first level, so a single
controller must handle the whole progression.
