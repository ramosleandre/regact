#!/bin/bash
# Benchmark driver: serve a model once via SimpleLM, then run one regact
# experiment PER TASK against it (one experiment dir per task, seeded naming).
# Used as the --client-command for ClusterControl's run_inference_job_ada.sh
# (venv already activated; cwd = the regact repo), like simplelm_regact.sh -
# which runs ONE experiment for a task list; this one gives each task its own
# experiments/bench_<date>/exp_<task>_<agent>-<model>_seed<seed>/ dir.
#
# Env in (from the pipeline): MODEL_PATH, MODEL_NAME, OUTPUT_DIR, SLURM_JOB_ID.
# Tunables (export before submitting; the launcher uses --export=ALL):
#   BENCH_DATE            default today (UTC) -> output_root experiments/bench_<date>
#   PROBLEM               default `arc_agi` (or `minigrid`)
#   TASK_NAMES            comma-separated, e.g. "ls20,vc33" (REQUIRED)
#   SEED                  default `0` (problem.seed; ignored by deterministic ARC)
#   AGENT                 default `alan`
#   SANDBOX               default `false`
#   WALLTIME_S            default `3600` (per-task wall-clock cap, seconds)
#   N_EPISODES            default `1` (controller eval episodes per submission)
#   CONTEXT_WINDOW        default `30000` (agent.args.context_window)
#   TOOL_CALL_FORMAT      default empty = native; e.g. `hermes` to force text format
#   CONCURRENCY           default `4`. Tasks run against the shared endpoint this many
#                         at a time (1 = sequential). SimpleLM serializes generation
#                         under a lock, so this does NOT batch decode; the win is that
#                         while one task holds the lock generating, the others run their
#                         off-GPU phases (tool exec, env steps, HTTP), so the GPU stops
#                         idling during any single task's tool work. VRAM-safe (one live
#                         KV cache at a time); 3-5 keeps the lock always contended.
#   SIMPLELM_TOOL_PARSER  default `universal`
#   BASE_URL              default empty = serve MODEL_PATH locally. Set to an
#                         already-running endpoint (e.g. the 2-node PP server's
#                         http://<head-node>:9876/v1) to skip serving entirely.
#   SERVE_VENV            default empty = the active venv's `simplelm`. Set to a
#                         venv path (e.g. $WORKDIR/venv_ada_3_11) to serve from
#                         its binary (flash_attn) while run_exp stays on the
#                         active venv - required on MI250 to avoid SDPA-math OOM.
set -uo pipefail

PORT=${PORT:-9876}
SERVE_VENV=${SERVE_VENV:-}
SERVE_BIN="${SERVE_VENV:+${SERVE_VENV}/bin/}simplelm"
BASE="${BASE_URL:-http://127.0.0.1:${PORT}/v1}"
LOG_DIR="${OUTPUT_DIR:-/tmp}"; mkdir -p "${LOG_DIR}"
SLOG="${LOG_DIR}/simplelm.${SLURM_JOB_ID:-local}.log"
TP=${SIMPLELM_TOOL_PARSER:-universal}
BENCH_DATE=${BENCH_DATE:-$(date -u +%F)}
PROBLEM=${PROBLEM:-arc_agi}
TASK_NAMES=${TASK_NAMES:?comma-separated task list required}
SEED=${SEED:-0}
AGENT=${AGENT:-alan}
SANDBOX=${SANDBOX:-false}
# Deployment policy: cap the per-task budget EVAL_MARGIN_S below the Slurm hard-kill so regact
# stops the task itself (gracefully) with time left for a final eval, instead of being SIGKILL'd
# mid-run. `auto`/`null` derive the budget from the job's Slurm end time; an explicit integer is
# honored as-is; off-Slurm (local) falls back to 3600s.
EVAL_MARGIN_S=${EVAL_MARGIN_S:-120}
WALLTIME_S=${WALLTIME_S:-auto}
if [ "$WALLTIME_S" = auto ] || [ "$WALLTIME_S" = null ]; then
    _end="${SLURM_JOB_END_TIME:-}"
    case "$_end" in
        ''|*[!0-9]*) _end=$(scontrol show job "${SLURM_JOB_ID:-x}" -o 2>/dev/null | sed -n 's/.*EndTime=\([0-9T:-]*\).*/\1/p')
                     [ -n "$_end" ] && _end=$(date -d "$_end" +%s 2>/dev/null);;
    esac
    if [ -n "$_end" ] && [ "$_end" -gt 0 ] 2>/dev/null; then
        WALLTIME_S=$(( _end - $(date +%s) - EVAL_MARGIN_S ))
    else
        WALLTIME_S=3600
    fi
    [ "$WALLTIME_S" -lt 60 ] && WALLTIME_S=60
fi
N_EPISODES=${N_EPISODES:-1}
CONTEXT_WINDOW=${CONTEXT_WINDOW:-30000}
TOOL_CALL_FORMAT=${TOOL_CALL_FORMAT:-}
CONCURRENCY=${CONCURRENCY:-4}
# Cross-cluster convention: set CLUSTER=adastra|jz to nest runs as
# <output_root>/<cluster>-<model>/<stamp>/<task>, so both clusters' results merge
# into one viz tree (the model column comes from config.json, so names must match).
CLUSTER=${CLUSTER:-}

PID=""
if [ -z "${BASE_URL:-}" ]; then
    echo "[bench] serve ${MODEL_NAME} via ${SERVE_BIN} (tool-parser=${TP}) — log ${SLOG}"
    "${SERVE_BIN}" serve --model-path "${MODEL_PATH}" --model-name "${MODEL_NAME}" \
        --tool-parser "${TP}" --host 127.0.0.1 --port "${PORT}" > "${SLOG}" 2>&1 &
    PID=$!
fi

# 60 min: a big MoE (e.g. Coder-Next ~159 GB) loads in ~26-30 min, and high job
# parallelism adds shared-FS I/O contention that pushes it past 30 min - so 1800 was a
# footgun. A crashed serve still fails fast via the process check below; this only
# lengthens the wait for a genuinely slow-but-alive load.
READY_TIMEOUT_S=${READY_TIMEOUT_S:-3600}
ready=0
for i in $(seq 1 $((READY_TIMEOUT_S / 10))); do
    if curl -sf "${BASE}/models" >/dev/null 2>&1; then
        ready=1; echo "[bench] server ready after ~$((i * 10))s"; break
    fi
    if [ -n "${PID}" ]; then
        kill -0 "${PID}" 2>/dev/null || { echo "[bench] server died"; tail -60 "${SLOG}"; exit 1; }
    fi
    sleep 10
done
[ "${ready}" = 1 ] || { echo "[bench] ${BASE} not ready in ${READY_TIMEOUT_S}s"; [ -n "${PID}" ] && { tail -60 "${SLOG}"; kill "${PID}"; } 2>/dev/null; exit 1; }

export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
EXTRA_ARGS=("agent.args.context_window=${CONTEXT_WINDOW}")
[ -n "${TOOL_CALL_FORMAT}" ] && EXTRA_ARGS+=("+agent.args.tool_call_format=${TOOL_CALL_FORMAT}")
# Per-run hydra overrides forwarded from the sbatch (max_output_tokens, escalated_max_tokens,
# tool_protocol, sandbox_opts.*, limits.*); space-separated, word-split into the arg array.
[ -n "${EXTRA_HYDRA:-}" ] && EXTRA_ARGS+=(${EXTRA_HYDRA})

IFS=',' read -ra TASKS <<< "${TASK_NAMES}"
WORK_DIR="$(mktemp -d)"  # per-task exit codes + unique Hydra run dirs (outside experiments/)

# One task's run_exp; exit code parked in WORK_DIR so the parent can tally after the wait.
# A distinct hydra.run.dir per task keeps concurrent invocations off Hydra's shared default
# outputs/<timestamp>/ dir; it lives outside experiments/ so it never looks like a run stamp.
run_task_bench() {
    local task="$1"
    local exp="${EXPERIMENT_NAME:-}"
    if [ -z "${exp}" ]; then
        if [ -n "${CLUSTER}" ]; then
            exp="${CLUSTER}-${MODEL_NAME}"  # tasks nest under one cluster-model dir
        else
            exp="exp_${task}_${AGENT}-${MODEL_NAME}_seed${SEED}"
        fi
    fi
    local oroot="${OUTPUT_ROOT:-experiments/bench_${BENCH_DATE}}"
    echo "[bench] start task=${task} -> ${oroot}/${exp}"
    python -m regact.run_exp \
        agent="${AGENT}" \
        agent.model="openai/${MODEL_NAME}" \
        agent.base_url="${BASE}" \
        agent.api_key="${OPENAI_API_KEY}" \
        problem="${PROBLEM}" \
        "problem.tasks=[${task}]" \
        problem.lifecycle=multi_instance \
        problem.seed="${SEED}" \
        controller.n_episodes="${N_EPISODES}" \
        limits.max_seconds_per_task="${WALLTIME_S}" \
        output_root="${oroot}" \
        experiment_name="${exp}" \
        sandbox="${SANDBOX}" \
        hydra.run.dir="${WORK_DIR}/hydra_${task}" \
        "${EXTRA_ARGS[@]}"
    local rc=$?
    echo "${rc}" > "${WORK_DIR}/${task}.rc"
    echo "[bench] done task=${task} exit=${rc}"
}

# Bounded parallelism: at most CONCURRENCY task jobs live at once. Count only OUR task
# PIDs (not the server), so the throttle is unaffected by the background SimpleLM process.
task_pids=()
_running() { local n=0 p; for p in "${task_pids[@]}"; do kill -0 "${p}" 2>/dev/null && n=$((n + 1)); done; echo "${n}"; }

echo "[bench] running ${#TASKS[@]} tasks, up to ${CONCURRENCY} concurrent against ${BASE}"
for task in "${TASKS[@]}"; do
    while [ "$(_running)" -ge "${CONCURRENCY}" ]; do sleep 2; done
    run_task_bench "${task}" &
    task_pids+=("$!")
done
for p in "${task_pids[@]}"; do wait "${p}"; done

failed=0
for task in "${TASKS[@]}"; do
    rc=$(cat "${WORK_DIR}/${task}.rc" 2>/dev/null || echo 1)  # missing marker = it died = failure
    [ "${rc}" -ne 0 ] && failed=$((failed + 1))
done
rm -rf "${WORK_DIR}"

if [ -n "${PID}" ]; then
    kill "${PID}" 2>/dev/null; sleep 2; kill -9 "${PID}" 2>/dev/null; wait 2>/dev/null || true
fi
echo "[bench] done: $((${#TASKS[@]} - failed))/${#TASKS[@]} tasks ran clean"
exit 0
