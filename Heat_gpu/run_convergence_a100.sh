#!/bin/bash
#SBATCH -J ftdma_gpu_conv
#SBATCH -p cas_v100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment etc

# ============================================================
#  Filtered_TDMA GPU convergence (NP=1,2,4 only — NP=8 has 2-day queue wait).
#  Two backends (pascal, filtered) × 3 decompositions × 4 grid sizes
#  = 24 runs in one job.
# ============================================================

set -e

module purge
module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_coll=^hcoll
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
export CUDA_LIBDIR="${NVHPC_ROOT}/cuda/lib64"

# --- Resolve project root --------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi
PROJ="$(pwd)"
mkdir -p log results

echo "=============================================="
echo " host    : $(hostname)"
echo " date    : $(date '+%F %T')"
echo " job_id  : ${SLURM_JOB_ID:-interactive}"
echo " gpus    : $(nvidia-smi -L 2>/dev/null | wc -l) visible"
echo " modules : $(module list 2>&1 | tail -n +2)"
echo " cwd     : $PROJ"
echo "=============================================="

# --- Build ----------------------------------------------------------------
echo "[build] USE_CUDA=1 CUDA_ARCH=70 make heat_gpu (V100)"
USE_CUDA=1 CUDA_ARCH=70 make heat_gpu

EXE="${PROJ}/build/bin/heat_gpu.out"
RUN_DIR="${PROJ}/Heat_gpu/inputs"
RESULT="${PROJ}/results/heat_gpu_convergence_${SLURM_JOB_ID:-local}.txt"

[ -x "${EXE}" ] || { echo "[ERROR] missing binary ${EXE}" >&2; exit 1; }
echo
echo "================================================================"
echo " Convergence sweep (rho=0.25, T_final=Tmax fixed, dt=dx^2)"
echo "                   2 backends × 4 decompositions × 4 N"
echo " EXE  : ${EXE}"
echo " RUN  : ${RUN_DIR}"
echo " LOG  : ${RESULT}"
echo "================================================================"
: > "${RESULT}"

for BACKEND in pascal filtered filtered_v2; do
    for LABEL in 1gpu 2gpu 4gpu; do
        case "${LABEL}" in
            1gpu) NP=1 ;;  2gpu) NP=2 ;;  4gpu) NP=4 ;;
        esac

        for N in 64 128 256 512; do
            INP="${RUN_DIR}/PARA_INPUT_${LABEL}_${N}.txt"
            if [ ! -f "${INP}" ]; then
                echo "[WARN] missing ${INP}, skip" | tee -a "${RESULT}"
                continue
            fi

            # Toggle backend in-place
            sed -i "s|^tdma_backend.*|tdma_backend = ${BACKEND}|" "${INP}"

            echo                                                                  | tee -a "${RESULT}"
            echo "=== backend=${BACKEND}  ${LABEL}  (NP=${NP})  N=${N} ==="       | tee -a "${RESULT}"
            T0=$(date +%s)
            mpirun --bind-to none -np ${NP} "${EXE}" "${INP}" 2>&1 | tee -a "${RESULT}"
            T1=$(date +%s)
            echo "[time] ${BACKEND} ${LABEL} N=${N}  this run $((T1-T0))s"        | tee -a "${RESULT}"
        done
    done
done

echo
echo "[done] all 32 runs complete, log: ${RESULT}"
