#!/bin/bash
# Filtered_v2-only scaling on H200 with register-resident kernels + 2-message MPI.
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/scaling_1comm/h200"
mkdir -p "${RES}"

echo "=== env on $(hostname)  [$(date '+%F %T')] ==="
[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }
echo "BIN=$(ls -lh ${BIN} | awk '{print $5, $9}')"
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

echo ""; echo "===== STRONG SCALING filtered_v2 np=2,4,8 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/strong_64x64x512_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/strong_256x256x2048_np${np}_filtered_v2_${rho}.txt"
done; done

echo ""; echo "===== WEAK SCALING filtered_v2 np=2,4,8 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case ${np} "${INP}/weak_128cube_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/weak_256cube_np${np}_filtered_v2_${rho}.txt"
    run_case ${np} "${INP}/weak_512cube_np${np}_filtered_v2_${rho}.txt"
done; done

echo ""; echo "===== REFINE SCALING filtered_v2 np=2 ====="
for inp in $(ls "${INP}"/refine_*_np2_filtered_v2_*.txt | sort); do
    tag=$(basename "${inp}" .txt)
    [ -f "${RES}/timing_${tag}.csv" ] && { echo "  [exists] ${tag}"; continue; }
    echo "---- ${tag}  [$(date +%H:%M:%S)] ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np 2 "${BIN}" "${inp}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]" || true
done

echo ""; echo "==== ALL DONE [$(date '+%F %T')]  results: ${RES} ===="
echo "Filtered CSVs: $(ls ${RES}/timing_*filtered*.csv 2>/dev/null | wc -l)"
