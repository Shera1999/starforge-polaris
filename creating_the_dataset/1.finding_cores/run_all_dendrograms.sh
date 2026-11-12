#!/usr/bin/env bash
# Run find_starforge_cores_dendrogram_v5.py (v5.2 defaults) on every snapshot,
# several at once. Existing outputs are skipped, so the run is resumable.
# Usage: ./run_all_dendrograms.sh [SNAPSHOT_DIR] [OUTPUT_ROOT] [N_PARALLEL]
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOT_DIR="${1:-/scratch/STARFORGE/M2e5_R30_Res585}"
OUTPUT_ROOT="${2:-${SCRIPT_DIR}/dendrogram_cores_v5.2}"
N_PARALLEL="${3:-8}"
PYTHON=/home/ilion/0/sjafaritabar/software/miniforge3/envs/starforge/bin/python
export SCRIPT_DIR OUTPUT_ROOT PYTHON

mkdir -p -- "${OUTPUT_ROOT}/logs"

run_one() {
    snapshot="$1"
    name="$(basename -- "${snapshot}" .hdf5)"
    if [[ -e "${OUTPUT_ROOT}/${name}" ]]; then
        echo "SKIP ${name}"
        return 0
    fi
    if "${PYTHON}" -u "${SCRIPT_DIR}/find_starforge_cores_dendrogram_v5.py" \
        --snapshot "${snapshot}" --output "${OUTPUT_ROOT}/${name}" --save-dendrogram \
        > "${OUTPUT_ROOT}/logs/${name}.log" 2>&1; then
        echo "OK ${name}"
    else
        echo "FAILED ${name} (see logs/${name}.log)"
    fi
}
export -f run_one

find "${SNAPSHOT_DIR}" -maxdepth 1 -type f -name 'snapshot_*.hdf5' -print0 | sort -z |
    xargs -0 -n 1 -P "${N_PARALLEL}" bash -c 'run_one "$0"'
