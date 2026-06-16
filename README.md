# regact

Agent-agnostic framework that drives a **code agent** (Alan Code in-process, or
the Claude Code CLI) to write **controllers** that play **games** (MiniGrid,
ARC-AGI-3). The agent reaches the environment only through an HTTP boundary and
never imports the game; new games, renderers, agents, and agent-built features
plug in without touching the core.

## Requirements

- Python **3.11–3.12** (not 3.13 — editable-install `.pth` footgun).

## Setup

```bash
make install          # editable install with dev tooling (ruff, mypy, pytest)
make check            # lint + typecheck + unit tests
```

Optional extras: `pip install -e ".[minigrid]"`, `".[arc]"`, `".[viz]"`.
The Alan Code backend is installed editable from its sibling repo:
`pip install -e ../Alan-Code-agent`.

## Common commands

```bash
make test             # unit tests (no LLM, no game)
make play GAME=...    # human-play an env (no agent)
make serve GAME=...   # launch the env server alone
make prompt GAME=...  # show the assembled prompt for a task
make viz              # launch the experiment visualizer
make run-exp ARGS=... # research run (Hydra overrides)
```

(Several commands light up as the build progresses; see the roadmap.)

## Layout

```
src/regact/        the package (agent, env, envclient, tools, eval, features,
                   problems, controllers, prompt, workspace, orchestration,
                   isolation, obs, session, config)
tests/             deterministic tests (scripted agent + fake env, no LLM)
```
