#!/bin/bash
#SBATCH -J ch550_h200_222_p
#SBATCH -p amd_h200nv_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --comment etc
#
# channel_gpu Re550 — PASCAL only, fresh start (no restart).
# np = 2 x 2 x 2 = 8 ranks (1 rank / GPU), H200 (amd_h200nv_8).
# Same conditions as sbatch_re550_filtered_h200_222.sh:
# stats accumulate from step 30000 through Timestepmax=60000.
# Separate output dirs (*_p_h200_222) so this does not clobber the
# filtered H200 run or any other job.
# Submit:  sbatch apps/channel_gpu/sbatch_re550_pascal_h200_222.sh
set -u

PROJ=/scratch/x3319a05/Filtered_TDMA
cd "${PROJ}/apps/channel_gpu"

# ---- GPU toolchain (nvhpc only) ----
source "${PROJ}/env_gpu.sh"

# ---- runtime env (CUDA-aware MPI / hpcx) ----
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n

BIN="${PROJ}/build_sm90/bin/channel_gpu.out"
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} missing (build first)"; exit 1; }

mkdir -p log statistics/re550_p_h200_222 instant/re550_p_h200_222 \
         restart_out/re550_p_h200_222 restart_in/re550_p_h200_222

rm -f instant/re550_p_h200_222/* statistics/re550_p_h200_222/* restart_out/re550_p_h200_222/*

echo "=== channel_gpu Re550 PASCAL only (np=2x2x2=8) on $(hostname)  [$(date '+%F %T')] ==="
echo "  bin: ${BIN}"
echo "  mpirun: $(which mpirun)"
nvidia-smi -L 2>&1 | head -8

t0=$(date +%s)
mpirun --bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll \
       -np 8 "${BIN}" input/PARA_INPUT_re550_pascal_h200_222.dat
rc_p=$?
t1=$(date +%s)
echo "[pascal done] rc=${rc_p}  wall=$((t1-t0))s  [$(date '+%F %T')]"
