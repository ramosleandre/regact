# Measuring what an action does

Each `make_env()` starts a NEW episode. To measure the effect of an action, keep ONE `env`
instance and compare `env.current()` before and after `env.step(...)` within that same episode,
rather than comparing observations taken from different `make_env()` calls.
