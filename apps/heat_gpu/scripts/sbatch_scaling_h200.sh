#!/bin/bash
#SBATCH --job-name=heatgpu-scaling
#SBATCH --comment=inhouse
#SBATCH --partition=amd_h200nv_8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/x3319a05/Filtered_TDMA
#SBATCH --output=/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_%j.out
# =============================================================================
#  Exclusive 8-GPU (H200) re-run of the heat_gpu strong/weak/refine scaling
#  sweep with the *profile* TDMA solve. Self-contained batch wrapper around
#  scripts/run_scaling_profile_h200.sh.
# =============================================================================
set -u
PROJ=/scratch/x3319a05/Filtered_TDMA
RESBASE="${PROJ}/apps/heat_gpu/results/scaling_profile/h200"

echo "############################################################"
echo " job   : ${SLURM_JOB_ID}  on $(hostname)"
echo " node  : exclusive, $(nvidia-smi -L 2>/dev/null | wc -l) GPUs visible"
echo " start : $(date '+%F %T')"
echo "############################################################"

# --- 1) Archive previous results (reversible 'delete') ----------------------
if [ -d "${RESBASE}" ]; then
    BAK="${RESBASE}.bak_$(date +%Y%m%d_%H%M%S)"
    mv "${RESBASE}" "${BAK}"
    echo "[archive] moved old results -> ${BAK}"
fi
mkdir -p "${RESBASE}/strong" "${RESBASE}/weak" "${RESBASE}/refine"

# --- 2) Run the strong/weak/refine sweep ------------------------------------
bash "${PROJ}/apps/heat_gpu/scripts/run_scaling_profile_h200.sh"

echo "==== batch DONE $(date '+%F %T') ===="
