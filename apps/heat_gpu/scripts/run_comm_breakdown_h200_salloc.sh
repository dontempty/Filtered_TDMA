#!/bin/bash
# Measure compute/comm breakdown for PaScaL + Filtered_v2 on H200.
# CSV gains 3 new columns: tdma_z_comm, tdma_y_comm, tdma_x_comm.
# Results saved to results/comm_breakdown/h200/ (separate from scaling_1comm).
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/comm_breakdown/h200"
mkdir -p "${RES}"

echo "=== Comm-breakdown run on $(hostname)  [$(date '+%F %T')] ==="
[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }
echo "BIN=$(ls -lh ${BIN} | awk '{print $5, $6, $7, $9}')"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -1

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)

run_case() {
    local np=$1 inp_file=$2
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  [skip] ${tag}"; return 0; }
    [ -f "${RES}/timing_${tag}.csv" ] && { echo "  [exists] ${tag}"; return 0; }
    echo "---- np=${np}  [$(date +%H:%M:%S)]  ${tag} ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]" || true
    return 0
}

# ---- Strong scaling np=2,4,8 (128^3 per-rank baseline) ----
echo ""; echo "===== STRONG SCALING pascal np=2,4,8 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_pascal_${rho}.txt"
done; done

echo ""; echo "===== STRONG SCALING filtered_v2 np=2,4,8 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_filtered_v2_${rho}.txt"
done; done

# ---- Large case np=8 for both solvers ----
echo ""; echo "===== LARGE CASE 256^3 np=8 ====="
for rho in rho025 rho040; do
    run_case 8 "${INP}/strong_256x256x2048_np8_pascal_${rho}.txt"
    run_case 8 "${INP}/strong_256x256x2048_np8_filtered_v2_${rho}.txt"
done

echo ""; echo "==== ALL DONE [$(date '+%F %T')]  results: ${RES} ===="
echo "CSVs written: $(ls ${RES}/timing_*.csv 2>/dev/null | wc -l)"
