#!/bin/bash
# Pascal-only scaling on H200 with merged alltoall_forward_3.
# Compares against existing filtered_v2 results in scaling_1comm/h200/.
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/scaling_1comm/h200"
mkdir -p "${RES}"

echo "=== env on $(hostname)  [$(date '+%F %T')] ==="
command -v mpirun >/dev/null || { echo "ERROR: mpirun not found"; exit 1; }
[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }
echo "BIN=$(ls -lh ${BIN} | awk '{print $5, $9}')"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -1

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)

run_case() {
    local label=$1 np=$2 inp_file=$3
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  [skip] ${tag}"; return 0; }
    # skip if already exists
    [ -f "${RES}/timing_${tag}.csv" ] && { echo "  [exists] ${tag}"; return 0; }
    echo "---- ${label}  np=${np}  [$(date +%H:%M:%S)]  ${tag} ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]|ERROR" || true
    return 0
}

echo ""; echo "===== STRONG SCALING pascal np=2,4,8 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case "strong 64x64x512"    ${np} "${INP}/strong_64x64x512_np${np}_pascal_${rho}.txt"
    run_case "strong 128x128x1024" ${np} "${INP}/strong_128x128x1024_np${np}_pascal_${rho}.txt"
    run_case "strong 256x256x2048" ${np} "${INP}/strong_256x256x2048_np${np}_pascal_${rho}.txt"
done; done

echo ""; echo "===== WEAK SCALING pascal np=2,4,8 ====="
for rho in rho025 rho040; do for np in 2 4 8; do
    run_case "weak 128cube/GPU" ${np} "${INP}/weak_128cube_np${np}_pascal_${rho}.txt"
    run_case "weak 256cube/GPU" ${np} "${INP}/weak_256cube_np${np}_pascal_${rho}.txt"
    run_case "weak 512cube/GPU" ${np} "${INP}/weak_512cube_np${np}_pascal_${rho}.txt"
done; done

echo ""; echo "===== REFINE SCALING pascal np=2 ====="
for inp in $(ls "${INP}"/refine_*_np2_pascal_*.txt | sort); do
    tag=$(basename "${inp}" .txt)
    [ -f "${RES}/timing_${tag}.csv" ] && { echo "  [exists] ${tag}"; continue; }
    echo "---- ${tag}  [$(date +%H:%M:%S)] ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np 2 "${BIN}" "${inp}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]|ERROR" || true
done

echo ""; echo "==== ALL DONE [$(date '+%F %T')]  results: ${RES} ===="
echo "Pascal CSVs: $(ls ${RES}/timing_*pascal*.csv 2>/dev/null | wc -l)"
