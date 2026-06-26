#!/bin/bash
# Strong + Weak + Refine GPU scaling on H200, using the EXISTING (boundary-fixed)
# build_sm90 binary (no rebuild). Launch with nvhpc loaded on the launcher:
#   module load nvhpc/25.11_cuda12
#   srun --jobid=<JID> --gres=gpu:8 -n1 --overlap bash run_scaling_all_norebuild.sh
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/scaling_1comm/h200"
mkdir -p "${RES}"
NODE=$(hostname)

echo "=== env on ${NODE} [$(date '+%F %T')] ==="
echo "  mpirun: $(which mpirun 2>&1)"
command -v mpirun >/dev/null || { echo "ERROR: mpirun not in PATH — load nvhpc before srun"; exit 1; }
[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }
echo "[bin] $(ls -lh ${BIN} | awk '{print $5,$NF}')"

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8

run_case() {
    local label=$1 np=$2 inp_file=$3
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  [skip missing] ${tag}"; return 0; }
    local cfg
    cfg=$(TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]|ERROR|error|fault" | tr '\n' ' ')
    echo "  [ok] np=${np} ${tag}  ${cfg}"
    return 0
}

echo ""; echo "===== STRONG SCALING  np=2,4,8  [$(date +%H:%M:%S)] ====="
for rho in rho025 rho040; do for be in pascal filtered_v2; do for np in 2 4 8; do
    run_case "strong" ${np} "${INP}/strong_64x64x512_np${np}_${be}_${rho}.txt"
    run_case "strong" ${np} "${INP}/strong_128x128x1024_np${np}_${be}_${rho}.txt"
    run_case "strong" ${np} "${INP}/strong_256x256x2048_np${np}_${be}_${rho}.txt"
done; done; done

echo ""; echo "===== WEAK SCALING  np=2,4,8  [$(date +%H:%M:%S)] ====="
for rho in rho025 rho040; do for be in pascal filtered_v2; do for np in 2 4 8; do
    run_case "weak" ${np} "${INP}/weak_128cube_np${np}_${be}_${rho}.txt"
    run_case "weak" ${np} "${INP}/weak_256cube_np${np}_${be}_${rho}.txt"
    run_case "weak" ${np} "${INP}/weak_512cube_np${np}_${be}_${rho}.txt"
done; done; done

echo ""; echo "===== REFINE SCALING  np=2  [$(date +%H:%M:%S)] ====="
for inp in $(ls "${INP}"/refine_*_np2_*.txt | sort); do
    run_case "refine" 2 "${inp}"
done

echo ""; echo "==== ALL DONE [$(date '+%F %T')]  results: ${RES} ===="
echo "CSV count: $(ls ${RES}/*.csv 2>/dev/null | wc -l)"
