#!/bin/bash
#SBATCH -J ch550_4gpu_a100
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.err
#SBATCH --time=05:00:00
#SBATCH --comment etc
#
# channel_gpu Re550 — PASCAL then FILTERED, fresh start (no restart),
# stats accumulate from step 20000 through Timestepmax=40000, checkpoint
# fileout every 20000 steps (instant field off).
# np = 2 x 1 x 2 = 4 ranks (x,z split only, no y split), A100 (amd_a100nv_8).
# Separate output dirs (*_4gpu) so this run does not clobber the
# concurrent 8-GPU (2x2x2, H200) job's results.
# Submit:  sbatch apps/channel_gpu/sbatch_re550_both_4gpu_a100.sh
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

BIN="${PROJ}/build_sm80/bin/channel_gpu.out"
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} missing (build first)"; exit 1; }

mkdir -p log statistics/re550_p_4gpu statistics/re550_f_4gpu \
         instant/re550_p_4gpu instant/re550_f_4gpu \
         restart_out/re550_p_4gpu restart_out/re550_f_4gpu \
         restart_in/re550_p_4gpu restart_in/re550_f_4gpu

# fresh start — no restart is read (ContinueFilein=0 in the input files),
# but clear any stale output from a previous attempt at these dirs.
rm -f instant/re550_p_4gpu/* statistics/re550_p_4gpu/* restart_out/re550_p_4gpu/*
rm -f instant/re550_f_4gpu/* statistics/re550_f_4gpu/* restart_out/re550_f_4gpu/*

echo "=== channel_gpu Re550 PASCAL+FILTERED (np=2x1x2=4, x/z split) on $(hostname)  [$(date '+%F %T')] ==="
echo "  bin: ${BIN}"
echo "  mpirun: $(which mpirun)"
nvidia-smi -L 2>&1 | head -8

# ---- [1] Pascal ----
echo ""
echo "=== [1] Pascal ==="
t0=$(date +%s)
mpirun --bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll \
       -np 4 "${BIN}" input/PARA_INPUT_re550_pascal_4gpu.dat
rc_p=$?
t1=$(date +%s)
echo "[pascal done] rc=${rc_p}  wall=$((t1-t0))s  [$(date '+%F %T')]"

# ---- [2] Filtered ----
echo ""
echo "=== [2] Filtered ==="
t0=$(date +%s)
mpirun --bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll \
       -np 4 "${BIN}" input/PARA_INPUT_re550_filtered_4gpu.dat
rc_f=$?
t1=$(date +%s)
echo "[filtered done] rc=${rc_f}  wall=$((t1-t0))s  [$(date '+%F %T')]"

echo ""
echo "==== ALL DONE [$(date '+%F %T')] ===="
echo "  pascal   rc=${rc_p}"
echo "  filtered rc=${rc_f}"
