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
WALLTIME_S=${WALLTIME_S:-3600}
N_EPISODES=${N_EPISODES:-1}
CONTEXT_WINDOW=${CONTEXT_WINDOW:-30000}
TOOL_CALL_FORMAT=${TOOL_CALL_FORMAT:-}

PID=""
if [ -z "${BASE_URL:-}" ]; then
    echo "[bench] serve ${MODEL_NAME} via ${SERVE_BIN} (tool-parser=${TP}) — log ${SLOG}"
    "${SERVE_BIN}" serve --model-path "${MODEL_PATH}" --model-name "${MODEL_NAME}" \
        --tool-parser "${TP}" --host 127.0.0.1 --port "${PORT}" > "${SLOG}" 2>&1 &
    PID=$!
fi

READY_TIMEOUT_S=${READY_TIMEOUT_S:-1800}
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

failed=0
IFS=',' read -ra TASKS <<< "${TASK_NAMES}"
for task in "${TASKS[@]}"; do
    exp="exp_${task}_${AGENT}-${MODEL_NAME}_seed${SEED}"
    echo "[bench] task=${task} -> experiments/bench_${BENCH_DATE}/${exp}"
    python -m regact.run_exp \
        agent="${AGENT}" \
        agent.model="openai/${MODEL_NAME}" \
        agent.base_url="${BASE}" \
        agent.api_key="${OPENAI_API_KEY}" \
        problem="${PROBLEM}" \
        "problem.tasks=[${task}]" \
        problem.lifecycle=multi_instance \
        problem.seed="${SEED}" \
        features.controller.n_episodes="${N_EPISODES}" \
        limits.max_seconds_per_task="${WALLTIME_S}" \
        output_root="experiments/bench_${BENCH_DATE}" \
        experiment_name="${exp}" \
        sandbox="${SANDBOX}" \
        "${EXTRA_ARGS[@]}"
    rc=$?
    echo "[bench] task=${task} exit=${rc}"
    [ "${rc}" -ne 0 ] && failed=$((failed + 1))
done

if [ -n "${PID}" ]; then
    kill "${PID}" 2>/dev/null; sleep 2; kill -9 "${PID}" 2>/dev/null; wait 2>/dev/null || true
fi
echo "[bench] done: $((${#TASKS[@]} - failed))/${#TASKS[@]} tasks ran clean"
exit 0
