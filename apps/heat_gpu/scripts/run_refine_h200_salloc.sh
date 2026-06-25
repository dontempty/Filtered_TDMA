#!/bin/bash
# Refine-only GPU scaling on H200 salloc (792356, gpu56).
# Binary already built at build_sm90/bin/heat_gpu.out.
# Launch from login node (nvhpc loaded) via:
#   srun --jobid=792356 --gres=gpu:8 -n1 --overlap bash scripts/run_refine_h200_salloc.sh
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build_sm90/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/scaling_1comm/h200"
mkdir -p "${RES}"

echo "=== env on $(hostname)  [$(date '+%F %T')] ==="
echo "  nvcc:   $(which nvcc 2>&1)"
echo "  mpirun: $(which mpirun 2>&1)"
command -v mpirun >/dev/null || { echo "ERROR: mpirun not in PATH"; exit 1; }
[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }
echo "[bin ok] $(ls -lh ${BIN} | awk '{print $5, $NF}')"

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"
NODE=$(hostname)
echo "=== GPUs on ${NODE} ==="; nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8

echo ""; echo "===== REFINE SCALING  np=2 ====="
for inp in $(ls "${INP}"/refine_*_np2_*.txt | sort); do
    tag=$(basename "${inp}" .txt)
    echo "---- ${tag}  [$(date +%H:%M:%S)] ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np 2 "${BIN}" "${inp}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]|ERROR|error|fault" || true
done

echo ""; echo "==== ALL DONE [$(date '+%F %T')]  results: ${RES} ===="
echo "Refine CSVs: $(ls ${RES}/timing_refine_*.csv 2>/dev/null | wc -l)"
