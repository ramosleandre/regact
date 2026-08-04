#!/bin/bash
# Serve a model via SimpleLM, wait for readiness, run a regact experiment against
# it, then kill the server. Used as the --client-command for ClusterControl's
# run_inference_job_ada.sh (venv already activated; cwd = the regact repo).
#
# Mirrors ClusterControl/scripts/clients/simplelm_gameagents.sh, adapted to regact.
#
# Env in (from the pipeline): MODEL_PATH, MODEL_NAME, EXP_NAME, OUTPUT_DIR, SLURM_JOB_ID.
# Tunables (export before submitting; the launcher uses --export=ALL):
#   SIMPLELM_TOOL_PARSER  default `universal` (per-family auto; `noop` = no tools)
#   TASK_NAMES            default `ls20`  (comma-separated, e.g. "ls20,vc33")
#   LIFECYCLE             default `multi_instance`  (or `single_instance`)
#   AGENT                 default `alan` (alancode in a sandboxable child process)
#   SANDBOX               default `false`. `true` = confine (auto backend) + deny egress;
#                         `apptainer` = confine forcing the apptainer backend (+ SIF below).
#   SIF                   apptainer image (.sif) — REQUIRED when SANDBOX=apptainer
#   WALLTIME_S            default `3000`  (per-game wall-clock cap, seconds)
#   OUTPUT_ROOT           default `experiments` -> regact writes experiments/<EXP_NAME>/<game>/,
#                         next to ClusterControl's experiments/<EXP_NAME>/logs/ (one dir per run)
set -uo pipefail

PORT=${PORT:-9876}
BASE="http://127.0.0.1:${PORT}/v1"
LOG_DIR="${OUTPUT_DIR:-/tmp}"; mkdir -p "${LOG_DIR}"
SLOG="${LOG_DIR}/simplelm.${SLURM_JOB_ID:-local}.log"
TP=${SIMPLELM_TOOL_PARSER:-universal}
TASK_NAMES=${TASK_NAMES:-ls20}
LIFECYCLE=${LIFECYCLE:-multi_instance}
SANDBOX=${SANDBOX:-false}
AGENT=${AGENT:-alan}
WALLTIME_S=${WALLTIME_S:-3000}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments}  # -> experiments/<EXP_NAME>/<game>/ (beside ClusterControl logs)

echo "[regact] serve ${MODEL_NAME} (tool-parser=${TP}) — log ${SLOG}"
simplelm serve --model-path "${MODEL_PATH}" --model-name "${MODEL_NAME}" \
    --tool-parser "${TP}" --host 127.0.0.1 --port "${PORT}" > "${SLOG}" 2>&1 &
PID=$!

# Big models load slowly from Lustre under HF transformers (32B ~25 min, no fast loader).
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1800}
ready=0
for i in $(seq 1 $((READY_TIMEOUT_S / 10))); do
    if curl -sf "${BASE}/models" >/dev/null 2>&1; then
        ready=1; echo "[regact] server ready after ~$((i * 10))s"; break
    fi
    kill -0 "${PID}" 2>/dev/null || { echo "[regact] server died"; tail -60 "${SLOG}"; exit 1; }
    sleep 10
done
[ "${ready}" = 1 ] || { echo "[regact] not ready in ${READY_TIMEOUT_S}s"; tail -60 "${SLOG}"; kill "${PID}" 2>/dev/null; exit 1; }

export AGENT_BASE_URL="${BASE}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
if [ "${SANDBOX}" = "apptainer" ]; then
    SANDBOX_ARGS=("sandbox=true" "+sandbox_opts.backend=apptainer"
                  "+sandbox_opts.image=${SIF:?SANDBOX=apptainer needs SIF=<image.sif>}")
else
    SANDBOX_ARGS=("sandbox=${SANDBOX}")  # true | false
fi

echo "[regact] run_exp problem=arc_agi tasks=[${TASK_NAMES}] agent=${AGENT} model=openai/${MODEL_NAME} sandbox=${SANDBOX}"
python -m regact.run_exp \
    agent="${AGENT}" \
    agent.model="openai/${MODEL_NAME}" \
    agent.base_url="${AGENT_BASE_URL}" \
    agent.api_key="${OPENAI_API_KEY}" \
    problem=arc_agi \
    "problem.tasks=[${TASK_NAMES}]" \
    problem.lifecycle="${LIFECYCLE}" \
    limits.max_seconds_per_task="${WALLTIME_S}" \
    output_root="${OUTPUT_ROOT}" \
    experiment_name="${EXP_NAME}" \
    "${SANDBOX_ARGS[@]}"
RC=$?
echo "[regact] run_exp exit=${RC}"

kill "${PID}" 2>/dev/null; sleep 2; kill -9 "${PID}" 2>/dev/null; wait 2>/dev/null || true
exit "${RC}"
