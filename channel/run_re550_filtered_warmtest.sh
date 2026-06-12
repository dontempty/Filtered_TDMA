#!/bin/bash
#SBATCH -J cwt_f
#SBATCH -p cpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=32
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --comment etc

# Warmtest: 5000 step run from restart, timing window step 2001-5000.
# Filtered_TDMA backend.
set -e

module purge
module load gcc/11.5.0 mpi/openmpi-4.1.8

if [ -n "${SLURM_SUBMIT_DIR}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
PROJ="$(realpath ..)"

EXE="${PROJ}/build/bin/channel.out"
[ -x "${EXE}" ] || { echo "[ERROR] missing ${EXE}" >&2; exit 1; }

mkdir -p log statistics/re590_f_warmtest instant/re590_f_warmtest \
         restart_out/re590_f_warmtest restart_in/re590_f

t0=$(date +%s)
mpirun --bind-to none -np 64 "${EXE}" PARA_INPUT_re550_filtered_warmtest.dat 2>&1
t1=$(date +%s)
echo
echo "[done]  filtered_warmtest wall-time (mpirun only) = $((t1 - t0)) s"
ls -la statistics/re590_f_warmtest/tdma_timing_rank*.csv 2>&1 | head
