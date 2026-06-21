"""Agent isolation, agnostic to any agent, environment, or feature.

Two independent, generic concerns (both take only paths, an argv, and tool-call
dicts -- never an agent/env/feature object):

- **Sandbox** (``runtime`` + ``contract`` + ``probe``): an OS sandbox wrapped around
  the agent's subprocess so paths outside its workdir (e.g. the game source) are
  absent from its filesystem view. ``wrap_argv`` is the single integration point; the
  rest of regact does not depend on which backend is used. The ``contract`` (R1-R6)
  is the same on every platform; only the enforcing mechanism differs (seatbelt /
  bwrap / landlock / apptainer / intrinsic). ``probe`` checks the contract identically
  on each platform.
- **Tool-call flagging** (``detection`` + ``policy``): a cheap scan of the agent's
  tool-call arguments that only logs/flags suspicious access for metrics; it never
  blocks. ``policy`` holds the shared forbidden-substring/import list it (and the
  deny-list) consumes. Missing an obfuscated attempt costs a log entry, not
  security -- the sandbox is what enforces confinement.
"""
