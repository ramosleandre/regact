# Running regact on Adastra (offline HPC)

End-to-end: from cloning regact to a working **ls20 multi-instance** run driven by
a **local model**, on Adastra. Adastra is AMD **ROCm**, and its **compute nodes
have no internet** — so the model is served locally by **SimpleLM** (OpenAI-compatible)
and regact drives it via the **Alan** agent, run in a sandboxable child process (the
cloud CLI agents codex/claude are unreachable there).

> **Source of truth for the cluster itself** is `ClusterControl/docs_adastra/setup.md`
> (validated). Do that first (SSH `ada`, `~/.bashrc.d/adastra.sh`, the `venv_ada`,
> model download). This page only adds **regact on top** and mirrors the validated
> job pattern (`pipeline.md` + `simplelm.md` + `simplelm_gameagents.sh`).

## 0. Prerequisites (from ClusterControl/docs_adastra/setup.md)
- `ssh ada` works (ControlMaster), `$WORKDIR`/`$SHAREDWORKDIR` resolve in a scripted shell.
- `~/.bashrc.d/adastra.sh` installed (module python/3.12.1, caches redirected, helpers).
- `venv_ada` built (`$WORKDIR/venv_ada`, torch 2.5.1+rocm6.2 + SimpleLM editable).

## 1. Clone + install regact (login node — it has internet)
```bash
ssh ada
cd $WORKDIR && git clone git@github.com:<your-org>/regact.git
source $WORKDIR/venv_ada/bin/activate
which python                                  # MUST be …/venv_ada/bin/python before installing
# regact's framework deps are light + CPU (the ARC engine); does NOT touch torch/ROCm.
pip install -e "$WORKDIR/regact[arc]"         # QUOTE the [arc] extra so the shell doesn't glob it
# The `arc-agi` PyPI pkg pulls `arcengine` automatically. If the extra didn't take (shell/proxy),
# install it directly — note the DASH (`pip install arc agi` with a space installs the wrong thing):
pip install arc-agi
python -c "import regact, arc_agi, arcengine; print('regact + arc_agi + arcengine OK')"   # MUST print before submitting
```
> Inode hygiene: this adds ~a few k files to the **shared** quota — check `usage_ada`.
> If a shared venv is in use, install regact into it instead of a personal one.

## 2. Pick a model (already on disk, shared)
```bash
ls $SHAREDWORKDIR/hf/        # the team's shared models
```
For a first "does it run" smoke, use a small **Adastra-tested** one:
`Qwen2.5-3B-Instruct` (it serves; it's weak at the games, but it validates the
plumbing). If a model is missing, download it on the **login** node:
`hf_download_ada <org/Model>` (see `simplelm.md`).

## 3. Launch: ls20 multi-instance (the smoke)
The client script [`scripts/adastra/simplelm_regact.sh`](../scripts/adastra/simplelm_regact.sh)
serves the model, waits for it, runs `python -m regact.run_exp`, then kills it.
Submit it through the validated pipeline (cwd = the regact repo):

```bash
ssh ada    # run from the LOGIN node; sbatch allocates a compute node
export TASK_NAMES=ls20 LIFECYCLE=multi_instance SANDBOX=false WALLTIME_S=3000
bash $WORKDIR/ClusterControl/scripts/pipelines/run_inference_job_ada.sh \
    --model Qwen2.5-3B-Instruct \
    --exp regact_ls20_smoke \
    --project-dir $WORKDIR/regact \
    --client-command 'bash $WORKDIR/regact/scripts/adastra/simplelm_regact.sh' \
    --gpus 8 --time 60
```
- **Single-quote** `--client-command` so `$MODEL_NAME`/`$AGENT_BASE_URL` resolve **on the node**.
- `--dry-run` first to inspect the sbatch line without spending a slot.
- Tunables are exported before the call (the launcher uses `--export=ALL`): `TASK_NAMES`,
  `LIFECYCLE`, `SANDBOX`, `WALLTIME_S`, `SIMPLELM_TOOL_PARSER`.

## 4. Did it work?
```bash
ls $WORKDIR/regact/experiments/adastra/regact_ls20_smoke/ls20/
#   logs/{transcript.jsonl, events.jsonl, experiment_state.json}
#   workdir/submissions/<n|final>/results.json   (+ video)
```
Sbatch logs: `$WORKDIR/ClusterControl/experiments/regact_ls20_smoke/logs/sbatch.<jid>.{out,err}`.
A successful run = the server became ready, `run_exp` exited 0, and a `results.json`
exists. Pull the experiment dir to your laptop and open it with `regact.viz.app` to inspect.

## 5. The sandbox

- **Smoke: `SANDBOX=false`.** The compute node is offline, so external egress is blocked
  intrinsically; this validates regact + the model + the game loop + the eval with the
  least moving parts. The only gap left open is the on-disk game files, which the OS
  sandbox closes.
- **Hardened: `SANDBOX=true`.** Adastra's compute nodes run **bwrap** with user namespaces
  enabled — validated: the R1–R6 contract holds, no `.sif` needed. The
  [`agent=alan`](agents.md) backend runs in a sandboxable child process, so `sandbox=true`
  confines it (and the netbridge keeps the local model reachable under `--unshare-net`).
  Validate first on a compute node:
  ```bash
  sbatch ... scripts/adastra/isolation_probe.sh   # expect CONTRACT HELD (see docs/sandboxing.md)
  ```

See **[Sandboxing](sandboxing.md)** for the full mechanism and the ready-to-submit probe
jobs (Adastra + Jean Zay).

## Notes / open points
- **Resident serving for sweeps**: don't reload the model per game — use the resident
  serve-job (`simplelm.md` §Resident serving) and fire trials at the live endpoint.
- **Tool parser**: `universal` by default (auto per family). `noop` returns no tool
  calls → the agent gets nothing. Override with `SIMPLELM_TOOL_PARSER=<parser>`.
