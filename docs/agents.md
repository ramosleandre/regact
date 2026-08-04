# Agents

The **agent** is the backend that writes the code. regact ships four:

| `agent=` | What it is | Needs |
|---|---|---|
| `scripted` | deterministic test backend, no LLM | nothing |
| `claude` | Claude Code CLI | the `claude` CLI, authenticated |
| `codex` | codex CLI | the `codex` CLI, authenticated |
| `alan` | Alan Code, run in a sandboxable child process against an OpenAI-compatible endpoint | `make install-agents` |

## Use an agent

Pick a backend by group name and override any field with a dotted path. The fields come
from `AgentConfig` — `model`, `base_url`, `api_key`, and a free-form `args` dict of
backend-specific CLI params.

```bash
# Claude, override the model and reasoning effort
make run ARGS="agent=claude agent.model=sonnet agent.args.effort=medium"

# codex
make run ARGS="agent=codex agent.args.reasoning_effort=high"

# Alan against a local server (SimpleLM/vLLM)
make run ARGS="agent=alan agent.model=openai/<MODEL> agent.base_url=http://127.0.0.1:9876/v1"
```

The config groups are in [`conf/agent/`](../src/regact/conf/agent/):

- **`claude.yaml`** — `model` (`opus`/`sonnet`/a full id), `args.permission_mode`,
  `args.effort` (`low…max`).
- **`codex.yaml`** — `model` (e.g. `gpt-5.5`), `args.reasoning_effort` (`low|medium|high`).
- **`alan.yaml`** — `model` (`openai/<name>`), `base_url`, `api_key`, and `args`
  (`permission_mode`, `max_output_tokens`, `context_window`, `backend`).
- **`scripted.yaml`** — just `name`.

### Installing the CLI agents

`claude` and `codex` are external programs — install and authenticate once:

```bash
npm install -g @anthropic-ai/claude-code   # then: claude   (login flow, cached)
npm install -g @openai/codex               # then: codex login   (or OPENAI_API_KEY)
```

regact passes **no** API key to the CLIs by default — they use their own cached auth.
Run `make agentcheck` to confirm a backend actually launches (bare and sandboxed).

## Add an agent

An agent implements the [`CodeAgent`](../src/regact/agent/base.py) ABC. For a CLI backend,
subclass [`_CliAgent`](../src/regact/agent/cli_agent.py) instead — it already implements
the whole subprocess loop, so you only fill three hooks. [`ClaudeAgent`](../src/regact/agent/claude_adapter.py)
is the canonical example.

**1. Subclass `_CliAgent`** and provide:

- `_command(self, message) -> (argv, stdin)` — the CLI invocation for one turn (argv list,
  never a shell string; optional stdin payload for secrets).
- `_parse_events(self, obj) -> list[AgentEvent]` — map one JSON line of the CLI's output
  to normalized events (`TextDelta`, `ToolCall`, `TurnComplete`, …).
- `_track_session(self, obj) -> None` (optional) — capture the native session id for
  resume.

Then declare the sandbox seams so the OS sandbox can confine it:

- `capabilities()` → a [`Capabilities`](../src/regact/agent/capabilities.py) (set
  `control_actions="client_cli"` for a subprocess agent — framework tools reach it over
  the workdir control CLI, not as in-process objects).
- `host_read_paths()` / `host_rw_paths()` / `host_egress_hosts()` — the host dirs and hosts
  this backend needs (install dirs, an isolated config home, its LLM host). Use
  `executable_paths("<cli>")` to resolve the binary's dirs.
- `launch_probe_argv()` → e.g. `["<cli>", "--version"]` so `agentcheck` can prove it starts.

**2. Register it** in two places:

- add a value to the `AgentName` enum in [`config/schema.py`](../src/regact/config/schema.py);
- add an `elif config.name is AgentName.<X>:` branch in `build_agent`
  ([`agent/base.py`](../src/regact/agent/base.py)) with a **lazy import** of your adapter.

**3. Add a config group** `conf/agent/<name>.yaml` with `name: <x>` and any default
`model` / `args`.

> **`native_tools` vs `client_cli`.** Only an **in-process** agent can execute framework
> tools as Python objects (`control_actions="native_tools"`, `executes_tools=True`). Every
> subprocess/CLI agent uses `client_cli`: it invokes SubmitSolution/ExitTask over the
> workdir's HTTP control channel. This keeps the loop provider-independent.
