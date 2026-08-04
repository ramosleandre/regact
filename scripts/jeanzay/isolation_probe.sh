#!/bin/bash
# Isolation diagnostics on a Jean Zay compute node (CPU, no GPU needed).
#
# Answers ONE question: which OS-sandbox path can confine agents on this cluster?
# Phase 1 inventories the node (bwrap? user namespaces? apptainer/singularity?);
# phase 2 runs regact's own diagnostics (doctor, conformance probe bare + sandboxed,
# agentcheck). Mirrors scripts/adastra/isolation_probe.sh with the JZ env recipe
# (ClusterControl/docs_jz/environments.md: no module purge — the python module is
# conda-backed and purging aborts the shell).
#
# Submit from a LOGIN node (see the runbook in the header of the sbatch command):
#   JOB_ROOT=$WORK/isoprobe_logs/$(date +%Y-%m-%d_%H-%M-%S); mkdir -p "$JOB_ROOT"
#   sbatch --account=imi@cpu --qos=qos_cpu-dev --hint=nomultithread \
#          --job-name=isoprobe --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:15:00 \
#          --output=$JOB_ROOT/isoprobe.%j.out --error=$JOB_ROOT/isoprobe.%j.err \
#          --export=ALL $WORK/regact/scripts/jeanzay/isolation_probe.sh
#
# Env (all optional):
#   REGACT_DIR   default $WORK/regact
#   VENV_PATH    default $SCRATCH/venv_regact
#   SIF          apptainer/singularity image (.sif) — when set, the sandboxed probe uses it
set -u

REGACT_DIR=${REGACT_DIR:-${WORK:?WORK not set}/regact}
VENV_PATH=${VENV_PATH:-${SCRATCH:?SCRATCH not set}/venv_regact}

section() { printf '\n===== %s =====\n' "$1"; }
try() { echo "\$ $*"; "$@" 2>&1; echo "[exit=$?]"; }

section "node"
try hostname
try uname -a
try cat /etc/os-release

section "backend inventory"
try which bwrap
command -v bwrap >/dev/null 2>&1 && try bwrap --version
try which apptainer
try which singularity
command -v apptainer >/dev/null 2>&1 && try apptainer --version
command -v singularity >/dev/null 2>&1 && try singularity --version
try which idrcontmgr   # IDRIS' container-image registration tool
if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
    echo "\$ module avail |& grep -iE 'apptainer|singular'"
    module avail 2>&1 | grep -iE 'apptainer|singular' || echo "(no apptainer/singularity module listed)"
fi

section "user namespaces"
try sysctl -n user.max_user_namespaces
try cat /proc/sys/kernel/unprivileged_userns_clone
try unshare --user --map-root-user true
command -v bwrap >/dev/null 2>&1 && try bwrap --ro-bind / / --unshare-user true

if [ -n "${SIF:-}" ]; then
    section "container smoke (SIF=${SIF})"
    if command -v apptainer >/dev/null 2>&1; then
        try apptainer exec --containall --no-home "${SIF}" true
    else
        try singularity exec --containall --no-home "${SIF}" true
    fi
fi

section "interpreter dynamic closure (what the sandbox must expose)"
REAL_PY=$(readlink -f "${VENV_PATH}/bin/python")
echo "real python: ${REAL_PY}"
try ldd "${REAL_PY}"

section "regact diagnostics"
# JZ env recipe (order matters; NEVER module purge here — conda-backed python module).
module load arch/h100 2>/dev/null || true
module load python/3.12.2 2>/dev/null || true
source "${VENV_PATH}/bin/activate" || { echo "FATAL: venv ${VENV_PATH} not activatable"; exit 1; }
cd "${REGACT_DIR}" || { echo "FATAL: ${REGACT_DIR} missing"; exit 1; }
export PYTHONPATH=${REGACT_DIR}/src
try python -m regact.doctor

section "probe: bare (unconfined baseline — VULNERABLE lines are EXPECTED here)"
try python -m regact.security.probe --no-egress

section "probe: sandboxed (what the detected backend actually defends)"
try python -m regact.security.probe --sandbox --no-egress ${SIF:+--image "$SIF"}

section "agentcheck (claude/codex expectedly absent on JZ)"
try python -m regact.agentcheck --all --verbose

section "done"
echo "Return the full .out/.err of this job."
