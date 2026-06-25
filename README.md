# regact

Agent-, game-, and feature-agnostic framework that drives a **code agent** (the
codex or Claude Code CLI, an in-process Alan agent, or a scripted test agent) to
write a **controller** — a pure `act(obs) -> action` policy — that plays a
**game** (ARC-AGI-3, MiniGrid). The agent reaches the environment only over an
HTTP boundary and never imports the game; new games, agents, and agent-built
features plug in behind small seams without touching the core.

## Requirements

- Python **3.11 or 3.12** (not 3.13).
- To run *sandboxed* on Linux: `bwrap` (bubblewrap) + unprivileged user
  namespaces; on HPC: Apptainer/Singularity; macOS dev uses the built-in
  `sandbox-exec`. **None** of these are needed to install or to run the tests.

## Install — plain `pip`, no `make` needed

```bash
python -m venv .venv && . .venv/bin/activate

# Clean / HPC / most reliable — copies the package into site-packages:
pip install .                      # + extras, e.g.  pip install ".[minigrid]"  ".[arc]"

# Local development — editable + lint/type/test tooling:
pip install -e ".[dev]"
```

After install the entry points run from **anywhere, with no `PYTHONPATH`**:

```bash
python -m regact.run_exp agent=codex problem=arc_agi 'task_names=[ls20]'
python -m regact.security.probe --sandbox          # check the OS sandbox on this host
python -m regact.viz.app --experiment experiments/<run>
```

> **macOS quirk:** if the repo is under `~/Desktop` and a bare `python -m regact…`
> says `No module named regact` after an *editable* install, that's a known macOS
> `.pth`/privacy interaction — use `pip install .` instead, or prefix
> `PYTHONPATH=src`. The test suite is unaffected (it injects `src` itself).

### Optional extras

| Extra | Brings | For |
|---|---|---|
| `dev` | ruff, mypy, pytest | development + the quality gate |
| `minigrid` | gymnasium, minigrid | the MiniGrid problem |
| `arc` | arc-agi | the ARC-AGI-3 problem |

The Alan in-process backend installs from its sibling repo: `pip install -e ../alancode`.

## Quality gate & tests (raw commands — `make` is only a convenience)

| Task | `pip`/raw command | `make` shortcut |
|---|---|---|
| Lint | `ruff check src tests` | `make lint` |
| Format + autofix | `ruff format src tests && ruff check --fix src tests` | `make format` |
| Type-check | `mypy src` | `make typecheck` |
| Unit tests (no LLM) | `pytest -m "not integration and not slow"` | `make test` |
| All tests | `pytest` | `make test-all` |
| Full gate | `ruff check src tests && mypy src && pytest` | `make check` |

## Run

```bash
# ARC-AGI-3 (local offline games), one game, with codex:
python -m regact.run_exp agent=codex problem=arc_agi 'task_names=[ls20]' agent.args.reasoning_effort=high

# MiniGrid with Claude:
python -m regact.run_exp agent=claude problem=minigrid
```

Outputs land under `experiments/<experiment_name>/<game>/`:
`logs/transcript.jsonl`, `logs/experiment_state.json`, and
`workdir/submissions/<n|final>/results.json` (+ a rollout video).

## Docs

| Doc | What |
|---|---|
| [docs/agents-setup.md](docs/agents-setup.md) | install + authenticate the CLI agents; full config reference |
| [docs/agent-isolation.md](docs/agent-isolation.md) | the anti-cheat / sandbox design (threat model, invariants R1–R6) |
| [docs/sandbox-testing.md](docs/sandbox-testing.md) | verify the sandbox per machine (the conformance probe) |
| [docs/debug_isolation_linux.md](docs/debug_isolation_linux.md) | **copy-paste commands to run on a Linux box**; paste the output back for analysis |
| [docs/contexte_isolation_state.md](docs/contexte_isolation_state.md) | **read first on a new machine** — current isolation state + the audited networking truth |

## Layout

```
src/regact/   the package (agent, env, envclient, tools, features, problems,
              controllers, prompt, workspace, orchestration, security, obs,
              session, viz, config)
tests/        deterministic tests (scripted agent + fake env, no LLM)
docs/         the docs above
```
