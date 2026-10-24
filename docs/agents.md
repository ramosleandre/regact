# Agents

The **agent** is the backend that writes the code. regact ships five:

| `agent=` | What it is | Needs |
|---|---|---|
| `scripted` | deterministic test backend, no LLM | nothing |
| `claude` | Claude Code CLI | the `claude` CLI, authenticated |
| `codex` | codex CLI | the `codex` CLI, authenticated |
| `antigravity` | Antigravity CLI (`agy`, Google) | the `agy` CLI, signed in, or `GEMINI_API_KEY` |
| `alan` | Alan Code, run in a sandboxable child process against an OpenAI-compatible endpoint | `make install-agents` |

Agent selection is independent of the [experiment protocol](protocols.md). The selected protocol supplies its instructions and framework commands. A backend's `tool_protocol` describes how it invokes tools; it does not select `vanilla`, `cwm` or `policy_search`.

## Use an agent

Pick a backend by group name and override any field with a dotted path. The fields come
from `AgentConfig` — `model`, `base_url`, `api_key`, and a free-form `args` dict of
backend-specific CLI params.

```bash
# Claude, override the model and reasoning effort
make run ARGS="agent=claude agent.model=sonnet agent.args.effort=medium"

# codex
make run ARGS="agent=codex agent.args.reasoning_effort=high"

# Antigravity, with the signed-in account (unsandboxed, see below)
make run ARGS="agent=antigravity agent.model=gemini-3.8-flash-high sandbox=false"

# Alan against a local server (SimpleLM/vLLM)
make run ARGS="agent=alan agent.model=openai/<MODEL> agent.base_url=http://127.0.0.1:9876/v1"
```

The config groups are in [`conf/agent/`](../src/regact/conf/agent/):

- **`claude.yaml`** — `model` (`opus`/`sonnet`/a full id), `args.permission_mode`,
  `args.effort` (`low…max`).
- **`codex.yaml`** — `model`, `args.reasoning_effort`, and `args.subagents_enabled`. Supported model/effort combinations depend on the installed CLI and account; the YAML is a preset, not a complete model catalogue.
- **`antigravity.yaml`** — `model` (an id from `agy models`), `args.auth` and `args.reasoning_effort`. regact gives agy its system prompt and a tool allowlist through an agent definition in a fresh per-task home: shell and file tools only (no web search, URL reader, browser, image generation or subagents; the file viewer only with `vision: true`, since it also shows images). Inside the sandbox that home is read-only except the folders agy writes, so the agent cannot widen its own tools. `auth: login` uses the signed-in account, whose token agy keeps in the desktop keyring; the sandbox cannot reach the keyring, so login runs need `sandbox=false` for now. `auth: api_key` uses `GEMINI_API_KEY` (billed as API usage) and runs sandboxed. agy's stream leaves out written file contents and the model's thinking; its own transcript, which has both, is kept in `logs/agy_transcripts/`.
- **`alan.yaml`** — `model` (`openai/<name>`), `base_url`, `api_key`, and `args`
  (`permission_mode`, `max_output_tokens`, `context_window`, `backend`).
- **`scripted.yaml`** — just `name`.
- **`alan_remote.yaml`** — the Alan backend driven through its HTTP remote interface, without model API calls. It selects `backend: scripted` and structured native tool calls (`tool_protocol: client_cli`), rather than asking Alan to extract tool calls from prose. Choose it with `agent=alan_remote`; it is a preset, not another agent implementation.

For a manual remote session, read the endpoint emitted by the Alan runner and use its `/api/respond` interface with structured `tool_calls`. The outer Regact tool/task budgets still apply while the remote session is waiting. This is distinct from `agent=scripted`, Regact's deterministic test backend. Exposing the local remote endpoint through the agent sandbox may require an appropriate local-network configuration; changing that does not grant submitted vanilla/CWM workers network access.

### Installing the CLI agents

`claude`, `codex` and `agy` are external programs — install and authenticate once:

```bash
npm install -g @anthropic-ai/claude-code   # then: claude   (login flow, cached)
npm install -g @openai/codex               # then: codex login   (or OPENAI_API_KEY)
curl -fsSL https://antigravity.google/cli/install.sh | bash   # then: agy   (sign-in flow)
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
  `tool_protocol="client_cli"` for a subprocess agent with native bash/file tools.
  Text-based Bash protocols are `"bash_block"`, `"hermes_xml"`, and `"glm"`; each teaches
  a different command format. All four reach framework tools over the workdir control
  CLI rather than as in-process objects).
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

> **`tool_protocol`.** The scripted backend uses `native` (in-process framework tools).
> Subprocess agents use `client_cli` (their own bash/file tools), `bash_block` (a fenced
> Bash command), `hermes_xml` (Qwen/Hermes-style markup), or `glm` (GLM-style markup).
> Subprocess agents reach the selected protocol's framework commands through the workdir's HTTP
> control channel: `framework/control.py` for policy search, `framework/commands.py` for vanilla/CWM. For Alan, select the dialect with `agent.args.tool_protocol`; the
> prompt and the backend parser must agree on that format.
