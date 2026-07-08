#!/bin/bash
#SBATCH -J ch550_h200_222_run1
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
# channel_gpu Re550 — FILTERED only, fresh start (no restart), fixed init seed.
# np = 2 x 2 x 2 = 8 ranks (1 rank / GPU), H200 (amd_h200nv_8).
# One of 3 IDENTICAL concurrent runs (run1/run2/run3) on 3 separate H200
# nodes, same config, same fixed seed -- to test whether independent H200
# node instances reproduce the same result (deterministic hardware/kernels)
# vs the earlier H200-vs-A100 comparison (different architecture).
# Stats accumulate from step 20000 through Timestepmax=40000.
# Separate output dirs (*_h200_222_run1) so this does not clobber any
# other job's results (including the other two run instances).
# Submit:  sbatch apps/channel_gpu/sbatch_re550_filtered_h200_222_run1.sh
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

mkdir -p log statistics/re550_f_h200_222_run1 instant/re550_f_h200_222_run1 \
         restart_out/re550_f_h200_222_run1 restart_in/re550_f_h200_222_run1

rm -f instant/re550_f_h200_222_run1/* statistics/re550_f_h200_222_run1/* restart_out/re550_f_h200_222_run1/*

echo "=== channel_gpu Re550 FILTERED run1 (np=2x2x2=8) on $(hostname)  [$(date '+%F %T')] ==="
echo "  bin: ${BIN}"
echo "  mpirun: $(which mpirun)"
nvidia-smi -L 2>&1 | head -8

t0=$(date +%s)
mpirun --bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll \
       -np 8 "${BIN}" input/PARA_INPUT_re550_filtered_h200_222_run1.dat
rc_f=$?
t1=$(date +%s)
echo "[filtered run1 done] rc=${rc_f}  wall=$((t1-t0))s  [$(date '+%F %T')]"
