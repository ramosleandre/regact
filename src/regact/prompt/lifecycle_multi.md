## This run: fresh episodes

Each `make_env()` and each `env.reset()` starts a **fresh, independent episode** — a new
layout from the same starting point. Your controller is re-created for every episode and
scored as the mean over many resets, so it must **generalize**: do not hard-code a fixed
action sequence or memorize a single layout. Aim for a policy that is robust across resets.

If your game has levels, every episode restarts from the first level, so a single
controller must handle the whole progression.
