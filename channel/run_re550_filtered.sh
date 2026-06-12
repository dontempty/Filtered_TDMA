#!/bin/bash
#SBATCH -J ch550_f
#SBATCH -p cpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --comment etc

# Re_tau ~ 550, Filtered_TDMA backend.
# 2 nodes x 32 ranks = 64 ranks (np = 4 x 4 x 4).
# Mesh 768 x 512 x 384  (Lx = 4 pi, Ly = 2 pi, H = 2).  Re_b = 10000
set -e

module purge
module load gcc/11.5.0 mpi/openmpi-4.1.8

if [ -n "${SLURM_SUBMIT_DIR}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
PROJ="$(realpath ..)"

EXE="${PROJ}/build/bin/channel.out"
[ -x "${EXE}" ] || { echo "[ERROR] missing ${EXE}" >&2; exit 1; }

mkdir -p log statistics/re590_f instant/re590_f \
         restart_out/re590_f restart_in/re590_f

t0=$(date +%s)
mpirun --bind-to none -np 64 "${EXE}" PARA_INPUT_re550_filtered.dat 2>&1
t1=$(date +%s)
echo
echo "[done]  filtered wall-time (mpirun only) = $((t1 - t0)) s"
ls -la statistics/re590_f/tdma_timing_rank*.csv 2>&1 | head
