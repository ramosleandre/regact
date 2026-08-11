#!/bin/bash
# Download an Adastra experiment run into local experiments/ so it can be presented
# and opened in `make viz`. Default = the FULL run (logs, config, the whole workdir
# incl. submissions/videos and the alan native transcript) - what you actually want
# to inspect a run. Set LIGHT=1 for the small slice (logs + config + solution.py +
# alan transcript only), when the multi-GB workdir is not needed.
#
#   scripts/fetch_adastra_run.sh <remote-leaf-dir> [local-name]
#   LIGHT=1 scripts/fetch_adastra_run.sh <remote-leaf-dir>
#
# <remote-leaf-dir> is the .../<task>/ dir under the experiment, either absolute or
# relative to $WORKDIR/regact/experiments on Adastra, e.g.:
#   bench_2026-08-08-newprompt/exp_ls20_alan-Qwen3-32B_seed0/2026-08-08_12-56-23/ls20
# Reuses the `ssh ada` ControlMaster (a human opens it once; keepalive holds it).
set -uo pipefail

REMOTE_ARG=${1:?remote run leaf dir required (…/<task>)}
LOCAL_NAME=${2:-}

# Resolve to an absolute remote path (accept absolute or experiments-relative).
if [[ "${REMOTE_ARG}" = /* ]]; then
    REMOTE="${REMOTE_ARG}"
else
    WORKDIR_REMOTE=$(ssh ada 'echo $WORKDIR') || { echo "cannot reach ada"; exit 1; }
    REMOTE="${WORKDIR_REMOTE}/regact/experiments/${REMOTE_ARG}"
fi

# The <task> leaf is kept as a subdir so `make viz EXP=experiments/adastra/<name>`
# works directly (the viewer expects <experiment>/<game>/logs/...). Default <name>
# is the exp dir (…/<exp>/<stamp>/<task> -> <exp>).
TASK=$(basename "${REMOTE_ARG}")
if [ -z "${LOCAL_NAME}" ]; then
    LOCAL_NAME=$(echo "${REMOTE_ARG}" | awk -F/ '{print $(NF-2)}')
fi
LOCAL="experiments/adastra/${LOCAL_NAME}"
DEST="${LOCAL}/${TASK}"
mkdir -p "${DEST}"

echo "[fetch] ${REMOTE}"
echo "[fetch]   -> ${DEST}  (${LIGHT:+LIGHT slice}${LIGHT:-full run})"
if [ -n "${LIGHT:-}" ]; then
    # Small slice: logs + config + the submitted controller + the alan transcript.
    rsync -az --prune-empty-dirs \
        --include='config.json' \
        --include='logs/' --include='logs/**' \
        --include='workdir/' \
        --include='workdir/solution.py' \
        --include='workdir/.alan/' --include='workdir/.alan/**/' \
        --include='workdir/.alan/**/transcript.jsonl' \
        --exclude='*' \
        "ada:${REMOTE}/" "${DEST}/" || { echo "[fetch] rsync failed"; exit 1; }
else
    # Full run - everything, so submissions/videos and the whole workdir are present.
    rsync -az "ada:${REMOTE}/" "${DEST}/" || { echo "[fetch] rsync failed"; exit 1; }
fi

echo "[fetch] done:"
find "${DEST}" -type f | sed "s|${LOCAL}/|  |"
echo "[fetch] view it:   make viz EXP=${LOCAL}"
echo "[fetch] alan log:  $(find "${DEST}" -path '*/.alan/*transcript.jsonl')"
