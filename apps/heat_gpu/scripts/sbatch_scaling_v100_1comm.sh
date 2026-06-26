#!/bin/bash
#SBATCH -J ftdma_v100_1c
#SBATCH -p cas_v100nv_8
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --comment etc
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_v100_1comm_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_v100_1comm_%j.out
# GPU strong+weak scaling np=2,4,8 — merged 1-comm FilteredTDMA vs PaScaL.
# Rebuilds heat_gpu (sm_70=V100) IN-JOB from current source into a per-arch
# BUILDDIR (build_sm70) so it never races the shared build/ or the a100 job.
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
BD="${PROJ}/build_sm70"
BIN="${BD}/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
RES="${PROJ}/apps/heat_gpu/results/scaling_1comm/v100"
mkdir -p "${RES}" "${PROJ}/apps/heat_gpu/log"

source /usr/share/Modules/init/bash 2>/dev/null || source /etc/profile.d/lmod.sh 2>/dev/null || true

# ---- in-job rebuild from current source (build modules: gcc+openmpi+nvhpc;
#      mpicxx resolves to nvhpc's hpcx) ----
echo "=== [BUILD] heat_gpu sm_70 → ${BD}  [$(date '+%F %T')] ==="
module purge >/dev/null 2>&1
module load gcc/15.2.0 mpi/openmpi-4.1.8 nvhpc/25.11_cuda12 >/dev/null 2>&1
( cd "${PROJ}" && rm -rf "${BD}" && USE_CUDA=1 CUDA_ARCH=70 make heat_gpu BUILDDIR="$(basename "${BD}")" ) 2>&1 | tail -6
[ -x "${BIN}" ] || { echo "ERROR: build failed — ${BIN} missing"; exit 1; }

# ---- runtime env: nvhpc ONLY (loading openmpi too shadows hpcx mpirun and
#      breaks multi-GPU CUDA-aware MPI) ----
module purge >/dev/null 2>&1
module load nvhpc/25.11_cuda12 >/dev/null 2>&1
export CUDA_LIBDIR=$NVHPC_ROOT/cuda/lib64
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll"
NODE=$(hostname)

echo "=== START $(date '+%F %T') host=${NODE} job=${SLURM_JOB_ID:-?} ==="
echo "BIN=${BIN}"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -8
[ -x "${BIN}" ] || { echo "ERROR: missing ${BIN}"; exit 1; }

run_case() {
    local label=$1 np=$2 inp_file=$3
    local tag; tag=$(basename "${inp_file}" .txt)
    [ -f "${inp_file}" ] || { echo "  [skip missing] ${tag}"; return 0; }
    echo "---- ${label}  np=${np}  [$(date +%H:%M:%S)]  ${tag} ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:8" -np ${np} "${BIN}" "${inp_file}" 2>&1 \
        | grep -E "rho\]|backend\]|Tmax\]|ERROR|error" || true
    return 0
}

echo ""; echo "===== STRONG SCALING  np=2,4,8 ====="
for rho in rho025 rho040; do for be in pascal filtered_v2; do for np in 2 4 8; do
    run_case "strong 64x64x512"    ${np} "${INP}/strong_64x64x512_np${np}_${be}_${rho}.txt"
    run_case "strong 128x128x1024" ${np} "${INP}/strong_128x128x1024_np${np}_${be}_${rho}.txt"
    run_case "strong 256x256x2048" ${np} "${INP}/strong_256x256x2048_np${np}_${be}_${rho}.txt"
done; done; done

echo ""; echo "===== WEAK SCALING  np=2,4,8 ====="
for rho in rho025 rho040; do for be in pascal filtered_v2; do for np in 2 4 8; do
    run_case "weak 128cube/GPU" ${np} "${INP}/weak_128cube_np${np}_${be}_${rho}.txt"
    run_case "weak 256cube/GPU" ${np} "${INP}/weak_256cube_np${np}_${be}_${rho}.txt"
    run_case "weak 512cube/GPU" ${np} "${INP}/weak_512cube_np${np}_${be}_${rho}.txt"
done; done; done

echo ""; echo "==== ALL DONE [$(date '+%F %T')]  results in ${RES} ===="
