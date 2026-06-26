#!/bin/bash
#SBATCH -J ftdma_strong_f
#SBATCH -p cas_v100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --comment etc

# ============================================================
#  Heat_gpu strong scaling — fixed problem size, vary NP.
#  Sizes: N = 512, 1024.  NP = 1, 2, 4, 8 (decomps: 1-1-1, 2-1-1, 2-2-1, 2-2-2).
#  TDMA backend: pascal (with the (128,1) block-dim optimization).
#  11 steps per run = 1 warmup (t_step==0) + 10 measured.
#  CSV → results/strong/timing_<N>_<npxnpynpz>_filtered_v2.csv
# ============================================================
set +e   # continue past per-case failures (e.g., OOM on N=1024 NP=1)

module purge
module load nvhpc/25.11_cuda12

# CUDA-aware MPI / UCX
export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib,uct

PROJ="/scratch/x3319a05/Filtered_TDMA"
EXE="${PROJ}/build/bin/heat_gpu.out"
RESULTS="${PROJ}/apps/heat_gpu/results/strong_filtered"
mkdir -p "${RESULTS}" log

[ -x "${EXE}" ] || { echo "[ERROR] missing ${EXE}" >&2; exit 1; }

echo "============================================================"
echo " host    : $(hostname)"
echo " date    : $(date '+%F %T')"
echo " gpus    : $(nvidia-smi -L 2>/dev/null | wc -l) visible"
echo "============================================================"

for N in 512 1024; do
    for NP in 1 2 4 8; do
        INP="${PROJ}/apps/heat_gpu/inputs/strong_${N}/PARA_INPUT_${NP}.txt"
        [ -f "${INP}" ] || { echo "[skip] missing ${INP}"; continue; }
        npxnpynpz=$(awk '/^npx /{x=$3}/^npy /{y=$3}/^npz /{z=$3;print x""y""z}' "${INP}")
        export TIMING_CSV="${RESULTS}/timing_${N}_${npxnpynpz}_filtered_v2.csv"
        echo
        echo "=== strong  N=${N}  NP=${NP}  npxnpynpz=${npxnpynpz} ==="
        T0=$(date +%s)
        mpirun --bind-to none -np ${NP} "${EXE}" "${INP}" 2>&1
        rc=$?
        T1=$(date +%s)
        echo "[wall] N=${N} NP=${NP}  $((T1-T0))s   rc=${rc}"
        unset TIMING_CSV
    done
done

echo
echo "[done] strong scaling CSVs at ${RESULTS}/"
ls -la "${RESULTS}"
