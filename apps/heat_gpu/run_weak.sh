#!/bin/bash
#SBATCH -J ftdma_weak
#SBATCH -p cas_v100nv_8
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --comment etc

# ============================================================
#  Heat_gpu weak scaling — fixed subdomain D per rank, vary NP.
#  Subdomain D = 128, 512.  NP = 1, 2, 4, 8.
#  Full mesh = D × (npx, npy, npz) — D-per-rank constant.
#  TDMA backend: pascal (with the (128,1) block-dim optimization).
#  11 steps per run = 1 warmup (t_step==0) + 10 measured.
#  CSV → results/weak/timing_<N>_<npxnpynpz>_pascal.csv
#       (N = global Nx for clarity; per-rank subdomain stays at D).
# ============================================================
set +e

module purge
module load nvhpc/25.11_cuda12

export UCX_MEMTYPE_CACHE=n
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib,uct

PROJ="/scratch/x3319a05/Filtered_TDMA"
EXE="${PROJ}/build/bin/heat_gpu.out"
RESULTS="${PROJ}/apps/heat_gpu/results/weak"
mkdir -p "${RESULTS}" log

[ -x "${EXE}" ] || { echo "[ERROR] missing ${EXE}" >&2; exit 1; }

echo "============================================================"
echo " host    : $(hostname)"
echo " date    : $(date '+%F %T')"
echo " gpus    : $(nvidia-smi -L 2>/dev/null | wc -l) visible"
echo "============================================================"

for D in 128 512; do
    for NP in 1 2 4 8; do
        INP="${PROJ}/apps/heat_gpu/inputs/weak_${D}/PARA_INPUT_${NP}.txt"
        [ -f "${INP}" ] || { echo "[skip] missing ${INP}"; continue; }
        nx_full=$(awk '/^nx /{print $3; exit}' "${INP}")
        npxnpynpz=$(awk '/^npx /{x=$3}/^npy /{y=$3}/^npz /{z=$3;print x""y""z}' "${INP}")
        export TIMING_CSV="${RESULTS}/timing_${nx_full}_${npxnpynpz}_pascal.csv"
        echo
        echo "=== weak  D=${D}  NP=${NP}  full=${nx_full} npxnpynpz=${npxnpynpz} ==="
        T0=$(date +%s)
        mpirun --bind-to none -np ${NP} "${EXE}" "${INP}" 2>&1
        rc=$?
        T1=$(date +%s)
        echo "[wall] D=${D} NP=${NP}  $((T1-T0))s   rc=${rc}"
        unset TIMING_CSV
    done
done

echo
echo "[done] weak scaling CSVs at ${RESULTS}/"
ls -la "${RESULTS}"
