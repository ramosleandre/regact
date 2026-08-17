# Keep one environment per experiment

Each `make_env()` call starts a NEW, potentially randomized episode. To measure what an
action does, keep ONE `env` instance and compare `env.current()` before and after
`env.step(...)`; never compare states taken from different `make_env()` calls, or you will
read the layout randomization as if it were an action's effect.
