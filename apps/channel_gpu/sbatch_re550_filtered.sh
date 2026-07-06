#!/bin/bash
#SBATCH -J ch550_f
#SBATCH -p amd_h200nv_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --comment etc
#
# channel_gpu Re550 — FILTERED, np = 2 x 2 x 2 = 8 ranks (1 rank / GPU), H200.
# Submit:  sbatch apps/channel_gpu/sbatch_re550_filtered.sh
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
INP="input/PARA_INPUT_re550_filtered.dat"
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} missing (build first)"; exit 1; }

mkdir -p log statistics/re550_f instant/re550_f restart_out/re550_f restart_in/re550_f
# fresh start — avoid mixing with previous re550_f outputs (append files etc.)
rm -f instant/re550_f/* statistics/re550_f/* restart_out/re550_f/*

echo "=== channel_gpu Re550 FILTERED (np=2x2x2=8) on $(hostname)  [$(date '+%F %T')] ==="
echo "  bin: ${BIN}"
echo "  mpirun: $(which mpirun)"
nvidia-smi -L 2>&1 | head -8

t0=$(date +%s)
mpirun --bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll \
       -np 8 "${BIN}" "${INP}"
rc=$?
t1=$(date +%s)
echo "[done] rc=${rc}  wall=$((t1-t0))s  [$(date '+%F %T')]"
