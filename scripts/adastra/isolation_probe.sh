#!/bin/bash
# Isolation diagnostics on an Adastra compute node (CPU, no GPU needed).
#
# Answers ONE question: which OS-sandbox path can confine agents on this cluster?
# Phase 1 inventories the node (bwrap? user namespaces?); phase 2 runs
# regact's own diagnostics (doctor, conformance probe bare + sandboxed, agentcheck).
#
# Submit from the LOGIN node (see docs/adastra-pipeline.md for the env setup):
#   JOB_ROOT=$WORKDIR/experiments_logs/$(date +%Y-%m-%d_%H-%M-%S)_isoprobe; mkdir -p "$JOB_ROOT"
#   sbatch --account=iso1996 --constraint=GENOA \
#          --job-name=isoprobe --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:15:00 \
#          --output=$JOB_ROOT/isoprobe.%j.out --error=$JOB_ROOT/isoprobe.%j.err \
#          --export=ALL $WORKDIR/regact/scripts/adastra/isolation_probe.sh
#
# Env (all optional):
#   REGACT_DIR   default $WORKDIR/regact
#   VENV_PATH    default $WORKDIR/venv_ada
set -u

REGACT_DIR=${REGACT_DIR:-${WORKDIR:?WORKDIR not set}/regact}
VENV_PATH=${VENV_PATH:-${WORKDIR}/venv_ada}

section() { printf '\n===== %s =====\n' "$1"; }
try() { echo "\$ $*"; "$@" 2>&1; echo "[exit=$?]"; }

section "node"
try hostname
try uname -a
try cat /etc/os-release

section "backend inventory"
try which bwrap
command -v bwrap >/dev/null 2>&1 && try bwrap --version
if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
fi

section "user namespaces"
try sysctl -n user.max_user_namespaces
try cat /proc/sys/kernel/unprivileged_userns_clone
try unshare --user --map-root-user true
command -v bwrap >/dev/null 2>&1 && try bwrap --ro-bind / / --unshare-user true


section "interpreter dynamic closure (what the sandbox must expose)"
REAL_PY=$(readlink -f "${VENV_PATH}/bin/python")
echo "real python: ${REAL_PY}"
try ldd "${REAL_PY}"

section "regact diagnostics"
module load python/3.12.1 2>/dev/null || true
source "${VENV_PATH}/bin/activate" || { echo "FATAL: venv ${VENV_PATH} not activatable"; exit 1; }
cd "${REGACT_DIR}" || { echo "FATAL: ${REGACT_DIR} missing"; exit 1; }
export PYTHONPATH=${REGACT_DIR}/src
try python -m regact.doctor

section "probe: bare (unconfined baseline — VULNERABLE lines are EXPECTED here)"
try python -m regact.security.probe --no-egress

section "probe: sandboxed (what the detected backend actually defends)"
try python -m regact.security.probe --sandbox --no-egress

section "agentcheck (claude/codex expectedly absent on Adastra)"
try python -m regact.agentcheck --all --verbose

section "done"
echo "Return the full .out/.err of this job."
