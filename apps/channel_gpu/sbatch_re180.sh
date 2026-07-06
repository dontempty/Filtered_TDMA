#!/bin/bash
#SBATCH -p amd_h200nv_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --comment etc
#
# channel_gpu Re180 — np = 2 x 2 x 2 = 8 ranks (1 rank / GPU), H200.
# Backend is the 1st arg. Submit BOTH at once:
#   sbatch -J ch180_f apps/channel_gpu/sbatch_re180.sh filtered
#   sbatch -J ch180_p apps/channel_gpu/sbatch_re180.sh pascal
set -u

BACKEND="${1:?usage: sbatch -J ch180_<f|p> sbatch_re180.sh <filtered|pascal>}"
case "${BACKEND}" in
  filtered) SUF=f ;;
  pascal)   SUF=p ;;
  *) echo "ERROR: backend must be 'filtered' or 'pascal'"; exit 1 ;;
esac

PROJ=/scratch/x3319a05/Filtered_TDMA
cd "${PROJ}/apps/channel_gpu"

source "${PROJ}/env_gpu.sh"
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n

BIN="${PROJ}/build_sm90/bin/channel_gpu.out"
INP="input/PARA_INPUT_re180_${BACKEND}.dat"
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} missing (build first)"; exit 1; }
[ -f "${INP}" ] || { echo "ERROR: ${INP} missing"; exit 1; }

mkdir -p log statistics/re180_${SUF} instant/re180_${SUF} \
         restart_out/re180_${SUF} restart_in/re180_${SUF}
# fresh start — avoid mixing with previous re180_${SUF} outputs
rm -f instant/re180_${SUF}/* statistics/re180_${SUF}/* restart_out/re180_${SUF}/*

echo "=== channel_gpu Re180 ${BACKEND} (np=2x2x2=8) on $(hostname)  [$(date '+%F %T')] ==="
echo "  bin: ${BIN}   input: ${INP}"
nvidia-smi -L 2>&1 | head -8

t0=$(date +%s)
mpirun --bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll \
       -np 8 "${BIN}" "${INP}"
rc=$?
echo "[done] backend=${BACKEND} rc=${rc} wall=$(( $(date +%s) - t0 ))s  [$(date '+%F %T')]"
