# Running with real code agents (Claude Code, codex)

regact drives a code agent headless in a sandboxed working directory. The two CLI
backends are external programs you install and authenticate once; regact then
spawns them per turn and streams their JSON output. (`scripted` needs nothing —
it is the deterministic test backend.)

## Claude Code

1. **Install** the CLI (Node 18+):
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude --version
   ```
2. **Authenticate** with your subscription login (no API key needed):
   ```bash
   claude            # opens the login flow once; credentials are cached
   ```
   regact passes **no** API key by default — it uses this cached subscription auth.
   To use an API endpoint instead, set `agent.base_url`/`agent.api_key` in the config.
3. **Smoke test** it works headless and emits stream-json:
   ```bash
   make debug D=block4_5_agent_smoke ARGS=claude     # or: PYTHONPATH=src python debug/block4_5_agent_smoke.py claude
   ```
   You should see normalized events printed (TextDelta, …, TurnComplete).

## codex

1. **Install** the CLI:
   ```bash
   npm install -g @openai/codex
   codex --version
   ```
2. **Authenticate** (`codex login`, or `OPENAI_API_KEY` in the environment).
3. **Smoke test**:
   ```bash
   make debug D=block4_5_agent_smoke ARGS=codex
   ```

## Per-CLI config (Hydra groups)

Each agent/problem is a Hydra config group, so you pick one by name and override
its fields:
- `conf/agent/{scripted,claude,codex}.yaml` — model + backend `args` (Claude:
  `permission_mode`, `effort`; codex: `reasoning_effort`, optional `sandbox`).
- `conf/problem/{minigrid,arc_agi}.yaml` — lifecycle, obs/info mode, `kwargs`.

Claude runs **headless** with `permission_mode: bypassPermissions` so it never
blocks on an interactive permission prompt (the `.claude/settings.json` deny-list
+ the HTTP boundary stay the confinement). Override per run if you want prompts.

## Launching a run

Research (Hydra; pick the `agent`/`problem` groups, override fields as needed):
```bash
# MiniGrid with Claude
PYTHONPATH=src python -m regact.run_exp agent=claude problem=minigrid

# ARC-AGI-3 (offline local games) with codex, one game, high reasoning
# NOTE: quote bracketed lists so zsh doesn't glob them.
PYTHONPATH=src python -m regact.run_exp agent=codex problem=arc_agi \
    'task_names=[ls20]' agent.args.reasoning_effort=high
```

Competition (Kaggle profile YAML):
```bash
PYTHONPATH=src python -m regact.run_kaggle --config src/regact/config/profile/competition.yaml \
    --games ls20 ft09 --parallel 2
```

Outputs land under `experiments/<experiment_name>/<game>/`:
`logs/transcript.jsonl`, `logs/experiment_state.json`, `workdir/submissions/<n|final>/results.json`.

## Config reference (all available values)

Pick groups with `agent=<name>` / `problem=<name>`; override any field with a
dotted path (`agent.args.effort=high`). Quote bracketed lists in zsh
(`'task_names=[ls20]'`). To add a key not in the base config, prefix `+`.

**`agent=`** — `scripted` (test, no LLM), `claude`, `codex`. (`alan` exists as a
backend but has no run group yet.)

**Claude** (`agent=claude`):
- `agent.model`: alias `sonnet` | `opus` | `haiku`, or a full id like
  `claude-opus-4-8` / `claude-sonnet-4-6` / `claude-haiku-4-5`. `null` = CLI default.
- `agent.args.permission_mode`: `bypassPermissions` (default; headless, no prompt)
  | `acceptEdits` | `default` | `plan` | `dontAsk` | `auto`.
- `agent.args.effort`: `low` | `medium` | `high` | `xhigh` | `max`.

**codex** (`agent=codex`):
- `agent.model`: a codex model id (e.g. `gpt-5.5`); see `codex --help` /
  `~/.codex/config.toml`. `null` = CLI default.
- `agent.args.reasoning_effort`: `low` | `medium` | `high`.
- `agent.args.sandbox` (optional): `read-only` | `workspace-write` |
  `danger-full-access`. **Unset** (default) uses `--dangerously-bypass` so the agent
  can reach the localhost env; setting a sandbox may block that.
- `agent.args.ask_for_approval` (only with `sandbox`): `never` (default) |
  `on-request` | `untrusted`.

**`problem=`** — `minigrid`, `arc_agi`.
- `problem.lifecycle`: `multi_instance` (fresh env per episode) | `single_instance`
  (one make per game, RESET = level reset; ARC).
- `problem.obs_mode`: `raw` (ascii/structured/vlm later).
- `problem.info_mode`: `informative` (full description) | `minimal` (discover by
  interaction).
- `problem.kwargs` — minigrid: `env_id` (e.g. `MiniGrid-Empty-5x5-v0`,
  `MiniGrid-DoorKey-5x5-v0`); arc_agi: `operation_mode: offline`,
  `environments_dir: environnement`.
- `task_names`: `[]` = all games; ARC games are the 25 under `environnement/`
  (`ls20`, `ft09`, `vc33`, `ar25`, …); MiniGrid uses the configured `env_id`.

**Run-level:**
- `features`: `[controller]` (the only registered feature so far).
- `execution`: `sequential` | `parallel`; `parallel_workers`: int (>1 with parallel).
- `limits.{keep_alive, max_moves, walltime_s, token_budget}`.
- `experiment_name`, `output_root` (outputs under `<output_root>/<experiment_name>/`).

See it composed without running: `python -m regact.run_exp agent=claude problem=arc_agi --cfg job`.

## Debugging a run

- **Agent not found**: the smoke script tells you if `claude`/`codex` is on PATH.
- **Nothing happens / hangs**: run one game (`task_names=[ls20]`), watch
  `logs/output.log` (human) and `logs/events.jsonl` (structured). The agent's own
  output is mirrored to stdout.
- **Agent can't reach the env**: check `workdir/framework/make_env.py` has the
  right baked URL, and that the run printed the env server port. For CLI agents the
  server is a real localhost port; for `scripted` it is in-process.
- **Submission errors**: read `workdir/submissions/<n>/results.json` — the
  `error`/`error_category` says whether it was the controller (`agent_solution`),
  the env (`env_runtime`), or an anti-cheat rejection (`anti-cheat: …`).
- **Reproduce eval without the agent**: write a controller into `workdir/solution.py`
  and call the control channel (`python workdir/framework/control.py SubmitSolution`)
  or drive `ControllerExecutor` directly in a script.
- **Codex event mapping** is best-effort (the `--json` schema isn't pinned). If
  events look wrong, capture a few stdout lines and adjust `codex_adapter._parse_events`.
