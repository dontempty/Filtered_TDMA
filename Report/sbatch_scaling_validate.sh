#!/bin/bash
#SBATCH -J ftdma_valid
#SBATCH -p cas_v100nv_8
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --comment etc
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/Report/scaling_validate_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/Report/scaling_validate_%j.out
# Quick validation of the scaling setup on a NON-exclusive node: confirms the
# pre-built sm_70 binary, the scaling inputs, the merged 1-comm distributed path
# (np>1), and TIMING_CSV writing all work — before the big exclusive jobs run.
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/apps/heat_gpu/bin_arch/heat_gpu_sm70.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/Report/scaling_validate_csv"
rm -rf "${RES}"; mkdir -p "${RES}"

source /usr/share/Modules/init/bash 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null || true
module purge >/dev/null 2>&1
module load nvhpc/25.11_cuda12 >/dev/null 2>&1
export CUDA_LIBDIR=$NVHPC_ROOT/cuda/lib64
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll"
NODE=$(hostname)

echo "=== VALIDATE $(date '+%F %T') host=${NODE} job=${SLURM_JOB_ID:-?} ==="
echo "BIN=${BIN}"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -4
[ -x "${BIN}" ] || { echo "FAIL: missing binary ${BIN}"; exit 1; }

run_case() {
    local np=$1 inp_file=$2
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  FAIL: missing input ${tag}"; return 1; }
    local csv="${RES}/timing_${tag}.csv"
    echo "---- np=${np}  ${tag} ----"
    TIMING_CSV="${csv}" \
        mpirun ${MPI_FLAGS} --host "${NODE}:4" -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]|ERROR|error|illegal" || true
    if [ -s "${csv}" ]; then
        echo "  OK CSV: $(wc -l < "${csv}") lines, head:"
        head -2 "${csv}" | sed 's/^/    /'
    else
        echo "  FAIL: no/empty CSV ${csv}"
    fi
}

# np=1 (no comm), np=2 and np=4 (distributed 1-comm path) — pascal vs filtered_v2
for be in pascal filtered_v2; do
  echo ""; echo "===== backend=${be} ====="
  for np in 1 2 4; do
    run_case ${np} "${INP}/strong_64x64x512_np${np}_${be}_rho025.txt"
    run_case ${np} "${INP}/strong_128x128x1024_np${np}_${be}_rho025.txt"
  done
  run_case 2 "${INP}/weak_128cube_np2_${be}_rho025.txt"
  run_case 4 "${INP}/weak_128cube_np4_${be}_rho025.txt"
done

echo ""; echo "=== CSV count: $(ls ${RES}/*.csv 2>/dev/null | wc -l) ==="
echo "==== VALIDATE DONE [$(date '+%F %T')] ===="
