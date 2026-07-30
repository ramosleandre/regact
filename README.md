# regact

**regact is a research framework for building agents that interact with an
environment.** It drives a *code-writing agent* (the Claude Code or codex CLI, an
in-process Alan agent, or a scripted test agent) that plays a **game** (ARC-AGI-3,
MiniGrid) by using a set of **features** — the always-on one being `controller`, which
has the agent write a pure `act(obs) -> action` policy and submit it.

Its point is **agnosticity**, so you can run many experiments by swapping parts: the
**agent**, the **environment** (a "problem"), and the **features** are all pluggable
behind small seams. The agent reaches the environment only over a localhost **HTTP
boundary** and never imports the game, so the score measures understanding, not
memorization or cheating.

## Install

Python **3.11 or 3.12** (not 3.13). No `make` needed.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                 # dev + tooling  (add ".[arc]" ".[minigrid]" for games)
```

Check the machine is ready (Python, agent CLIs, sandbox, game extras):

```bash
make doctor
```

> **macOS quirk:** if a bare `python -m regact…` says `No module named regact` after an
> editable install (a known macOS `.pth` interaction under `~/Desktop`), prefix
> `PYTHONPATH=src` — the `make` targets already do this for you.

## Run

One command. Compose the four axes as Hydra groups (`agent`, `problem`, `features`,
`experiment`) and override any field on the CLI:

```bash
# 1) fastest end-to-end, no LLM and no game library (scripted agent, one game):
make run ARGS="experiment=dev"

# 2) a real ARC-AGI-3 run with a coding CLI (anti-cheat on, records a video):
make run ARGS="experiment=research agent=codex 'problem.tasks=[ls20]'"

# 3) MiniGrid with Claude:
make run ARGS="agent=claude problem=minigrid"
```

Then open the visualizer on the run (conversation, metrics, submissions, videos):

```bash
make viz EXP=experiments/<name>/latest
```

Each run gets its own timestamped directory, so re-running a name never overwrites the
previous run: `experiments/<name>/<timestamp>/<game>/` holds `logs/transcript.jsonl`,
`logs/experiment_state.json`, and `workdir/submissions/<n|final>/results.json` (+ a
video). `experiments/<name>/latest` always points at the most recent run.

## Configure (Hydra)

Everything is a Hydra group under [`src/regact/conf/`](src/regact/conf/) — one file per option:

| To configure a… | Edit / add a file in | Select on the CLI |
|---|---|---|
| **agent** | `conf/agent/` (`claude`, `codex`, `alan`, `scripted`) | `agent=claude` |
| **environment** (problem) | `conf/problem/` (`arc_agi`, `minigrid`) | `problem=arc_agi` |
| **features** (a set) | `conf/features/` (`controller`, `cwm`) | `features=cwm` |
| **experiment** (a whole profile) | `conf/experiment/` (`dev`, `research`, `competition`) | `experiment=research` |

A run loads a **map of features** — `controller` is the always-on base, but you can
stack more (`features={controller: {}, cwm: {}}`); the bootstrap, prompt, tools, env
wrapping, and teardown assemble themselves from that set. Each feature **owns its
knobs** (`features.cwm.max_tested_transitions_per_verify=500`).

Hyperparameters (single- vs multi-instance lifecycle, limits, sandbox, `shadow_replay`…)
are plain fields you set inline (`problem.lifecycle=single_instance limits.walltime_s=3600`)
or bundle into an `experiment` profile. The typed schema is
[`config/schema.py`](src/regact/config/schema.py); both front-ends
(`run_exp` via Hydra, `run_kaggle` via a YAML profile) build the same `RunConfig`.

## Commands (`make help`)

```
make help        # this list, grouped by category
make doctor      # is this machine ready to run regact?
make check       # the quality gate: lint + typecheck + unit tests
make run         # a research run (ARGS="...")
make viz         # the experiment visualizer (EXP=...)
make probe       # verify the OS sandbox on this host
```

---

## Architecture

**One keep-alive loop** wrapped around **three swappable plugins** meeting at an **HTTP
wall**. The loop sends a message to the agent, consumes a normalized `AgentEvent` stream,
runs any framework tool the agent calls, re-injects the result, and repeats until the
agent exits, a limit trips, or an error stops it.

```
                        orchestration/task.py   ← the one wiring hub
                        (joins the 3 by name + Capabilities, never by concrete type)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                          ▼
   ┌─────────┐              ┌───────────┐              ┌──────────┐
   │  AGENT  │              │  PROBLEM  │══ HTTP wall ═│ FEATURE  │
   │CodeAgent│              │(the env)  │  Obs (JSON)  │ Feature  │
   └────┬────┘              └───────────┘              └────┬─────┘
        │  AgentEvent stream                                │ Tool / Hook
        ▼                                                   ▼
   ┌──────────────────────── orchestration/loop.py ───────────────────┐
   │  send → consume events → run framework tools → inject → repeat    │
   │  agnostic: knows only agents / tools / hooks / limits / writers   │
   └───────────────────────────────────────────────────────────────────┘
```

### The three abstractions

Each is an ABC behind a lazy `string → factory` registry, so config names only
strings/enums — never a concrete class. To add one, drop a file and register a name;
the core is untouched.

| Axis | ABC | Defined in | Add one by |
|---|---|---|---|
| **Agent** — who writes the code | `CodeAgent` | [agent/base.py](src/regact/agent/base.py) | `AgentName` enum + `build_agent` branch + adapter + `conf/agent/*.yaml` |
| **Problem** — the environment (iterates its tasks) | `BaseProblem` | [problems/base.py](src/regact/problems/base.py) | `problems/<game>/problem.py` + one `_load_builtins()` line + `conf/problem/*.yaml` |
| **Feature** — an agent-built capability | `Feature` | [features/base.py](src/regact/features/base.py) | `features/<name>.py` + one `_load_builtins()` line + `conf/features/*.yaml` |

The agent is driven only through the `CodeAgent` ABC, described by a data-only
`Capabilities` descriptor and a normalized `AgentEvent` union — so the loop never
inspects a backend's type, and tool routing degrades on *data*, not on which adapter is
loaded.

### What a feature is

A **feature** is a self-contained capability a run turns on. It bundles four things,
and the core assembles a run from whatever set of features is loaded — the loop never
names a specific feature. The contract is [features/base.py](src/regact/features/base.py):

| A feature provides… | Method | Assembled into |
|---|---|---|
| workdir files | `templates(ctx)` | scaffolded into the agent's workdir |
| a prompt fragment | `prompt_fragment(ctx)` | appended to the agent's first message |
| tools | `tools(deps)` | tools the agent can call (run by the loop or over `/control`) |
| teardown hooks | `hooks(deps)` | framework work fired at a phase (e.g. re-score at the end) |
| an env wrapper (optional) | `env_wrapper(ctx)` | applied around the server-side env, in `features:` list order |

`templates`/`prompt_fragment`/`env_wrapper` take a static `FeatureContext`; `tools`/`hooks`
take a runtime `RunDeps` (the run's `EnvClient`, state, paths…). Built-in features:
[features/controller.py](src/regact/features/controller.py) — read it as the worked
example: it ships `base_controller.py` + a `solution.py` stub (templates), explains the
`act(obs) -> action` contract (prompt, in `features/prompts/controller.md`), provides the
`SubmitSolution` + `ExitTask` tools, and registers a teardown hook that re-scores the
final `solution.py` — and [features/cwm.py](src/regact/features/cwm.py), the Code World
Model: it records every env transition into `workdir/data/transitions.jsonl` (an env
wrapper) and scaffolds `world_model/` with a self-contained `verify.py` the agent runs
to measure its model's coherence (design: `context/cwm_v2.md`). A new feature is one
file next to these plus `register_feature(...)`.

### The HTTP wall (why the agent never imports the game)

[env/server.py](src/regact/env/server.py) ↔ [envclient/client.py](src/regact/envclient/client.py):
only serialized JSON crosses — an opaque action out, an `Obs` DTO
([envclient/obs.py](src/regact/envclient/obs.py): `frame`/`reward`/`is_done`/`available_actions`/`info`)
back. A server-side `ObsRenderer` flattens the native observation into `Obs`; the agent
cannot override it. The game engine and its answer key run in a separate process and are
never in the agent's filesystem view — **prevention by absence**.

### Trusted vs untrusted (why a controller cannot fake its score)

The agent writes `solution.py`, which could otherwise compute — and lie about — its own
score. regact splits evaluation in two ([controllers/executor.py](src/regact/controllers/executor.py)):

- **Untrusted** (`run_episodes_raw`) — the agent's controller runs in a sandboxed
  subprocess and only **records what the env returned** (observations, actions). It never
  computes a score and has no access to the scoring function.
- **Trusted** (`score_episodes` / `replay_and_score`) — the **orchestrator**, outside the
  sandbox, applies the problem's metric to those recordings. With `shadow_replay`, it
  re-runs the recorded actions on a *fresh trusted env* and re-derives the score.

So the score is computed by code the agent didn't write, on an env it doesn't control —
neutralizing both faked metrics and memorized action replays.

### Isolation, woven through — not central

[security/](src/regact/security/) is a stdlib-only leaf (imports no other regact module).
The OS sandbox (`runtime.wrap_argv`: none / seatbelt / bwrap / apptainer) + an egress
allow-list proxy are **enforced**; a detection "camera" (`detection.py` + `policy.py`) is
**advisory** — it only counts and logs suspicious tool calls, never blocks. The R1–R6
isolation contract is checked per host by the conformance probe (`make probe`).

## Docs

| Doc | What |
|---|---|
| [docs/agents-setup.md](docs/agents-setup.md) | install + authenticate the CLI agents; full config reference |
| [docs/agent-isolation.md](docs/agent-isolation.md) | the anti-cheat / sandbox design (threat model, invariants R1–R6) |
| [docs/sandbox-testing.md](docs/sandbox-testing.md) | verify the sandbox per machine (the conformance probe) |
| [docs/contexte_isolation_state.md](docs/contexte_isolation_state.md) | **read first on a new machine** — current isolation state |

## Layout

```
src/regact/
  orchestration/  the conductor: loop (agnostic keep-alive) + task (the wiring hub)
  agent/          the Agent plugin: CodeAgent ABC, Capabilities, AgentEvent, adapters
  problems/       the Problem plugin: BaseProblem ABC + arc_agi, minigrid
  features/       the Feature plugin: Feature ABC + the always-on controller
  controllers/    the untrusted/trusted eval split (executor, runner, summary)
  env/ envclient/ the HTTP wall: server + wrapped env / renderer  ·  client + Obs
  prompt/ workspace/ tools/   system prompt · workdir bootstrap · framework tools
  security/       anti-cheat & sandbox: runtime wrap, egress proxy, R1–R6 probe
  obs/ viz/ session/  transcript+logs · dashboard · run state
  conf/ config/   Hydra config groups · the typed RunConfig schema + loader
tests/            deterministic tests (scripted agent + fake env, no LLM)
```
