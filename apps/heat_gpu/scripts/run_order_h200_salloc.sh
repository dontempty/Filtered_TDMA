#!/bin/bash
# Convergence verification: merged alltoall_forward_3 must give same L2 errors as np=1.
# Tests pascal + filtered_v2 with np=1,2,4 on 64/128/256 grids.
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
REFDIR="${PROJ}/apps/heat_gpu/results/order_v100"
RES="${PROJ}/apps/heat_gpu/results/order_h200"
mkdir -p "${RES}"

echo "=== env on $(hostname)  [$(date '+%F %T')] ==="
command -v mpirun >/dev/null || { echo "ERROR: mpirun not found"; exit 1; }
[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }
echo "BIN=$(ls -lh ${BIN} | awk '{print $5, $9}')"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -1

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
NODE=$(hostname)
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"

run_one() {
    local backend=$1 sz=$2 np=$3
    local inp="${REFDIR}/in_${backend}_${sz}.txt"
    [ -f "${inp}" ] || { echo "  [skip missing] ${inp}"; return; }
    local tmpf="/tmp/order_${backend}_${sz}_np${np}.txt"
    sed "s/npz = 1/npz = ${np}/" "${inp}" > "${tmpf}"
    echo -n "  ${backend} ${sz}^3 np=${np}: "
    mpirun ${MPI_FLAGS} --host "${NODE}:8" -np ${np} "${BIN}" "${tmpf}" 2>&1 \
        | grep -E "L2 error" | awk '{print $NF}'
}

echo ""
echo "===== Pascal convergence (np=1/2/4) ====="
for sz in 64 128 256; do
    for np in 1 2 4; do
        run_one pascal ${sz} ${np}
    done
    echo ""
done

echo "===== Filtered_v2 convergence (np=1/2/4) ====="
for sz in 64 128 256; do
    for np in 1 2 4; do
        run_one filtered_v2 ${sz} ${np}
    done
    echo ""
done

echo "==== DONE [$(date '+%F %T')] ===="
