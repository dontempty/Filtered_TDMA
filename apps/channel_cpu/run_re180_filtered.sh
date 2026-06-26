#!/bin/bash
#SBATCH -J ch180_f
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --comment etc

# Re_tau ~ 180 channel, Filtered_TDMA backend.  1 node x 32 ranks (np = 4x4x2).
# Mesh 256^3.  Run 30000 steps; stats accumulate 10000->30000; stats & instant
# fields output every 10000 steps.  Shared node (not exclusive).
set -e
module purge
module load gcc/11.5.0 mpi/openmpi-4.1.8

if [ -n "${SLURM_SUBMIT_DIR}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
PROJ="$(realpath ../..)"
EXE="${PROJ}/build/bin/channel.out"
[ -x "${EXE}" ] || { echo "[ERROR] missing ${EXE}" >&2; exit 1; }

mkdir -p log statistics/re180_f instant/re180_f restart_out/re180_f restart_in/re180_f

echo "=== START $(date '+%F %T')  host=$(hostname)  job=${SLURM_JOB_ID:-?} ==="
echo "EXE=${EXE}  input=input/PARA_INPUT_re180_filtered.dat  np=32"
t0=$(date +%s)
mpirun --bind-to none -np 32 "${EXE}" input/PARA_INPUT_re180_filtered.dat 2>&1
t1=$(date +%s)
echo "[done] filtered wall-time (mpirun only) = $((t1 - t0)) s"
