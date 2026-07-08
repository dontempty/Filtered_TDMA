#!/bin/bash
#SBATCH -J ch550_4gpu_v100pcie
#SBATCH -p cas_v100nv_8
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/channel_gpu/log/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --comment etc
#
# channel_gpu Re550 — FILTERED only, fresh start (no restart).
# Retry of the V100-SXM (cas_v100_4) 4-GPU run that hit its 3h time limit
# at step ~23820/40000 — same np=2x1x2=4 config, now on V100 PCIe
# (cas_v100nv_8, gpu03 had 7/8 GPUs free), 6h budget this time.
# Separate output dirs (*_4gpu_v100pcie) so this does not clobber the
# incomplete V100-SXM run's partial output, or the H200/A100/8gpu-v100pcie jobs.
# Submit:  sbatch apps/channel_gpu/sbatch_re550_filtered_4gpu_v100pcie.sh
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

BIN="${PROJ}/build_sm70/bin/channel_gpu.out"
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} missing (build first)"; exit 1; }

mkdir -p log statistics/re550_f_4gpu_v100pcie instant/re550_f_4gpu_v100pcie \
         restart_out/re550_f_4gpu_v100pcie restart_in/re550_f_4gpu_v100pcie

rm -f instant/re550_f_4gpu_v100pcie/* statistics/re550_f_4gpu_v100pcie/* restart_out/re550_f_4gpu_v100pcie/*

echo "=== channel_gpu Re550 FILTERED only (np=2x1x2=4, x/z split) on $(hostname)  [$(date '+%F %T')] ==="
echo "  bin: ${BIN}"
echo "  mpirun: $(which mpirun)"
nvidia-smi -L 2>&1 | head -8

t0=$(date +%s)
mpirun --bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll \
       -np 4 "${BIN}" input/PARA_INPUT_re550_filtered_4gpu_v100pcie.dat
rc_f=$?
t1=$(date +%s)
echo "[filtered done] rc=${rc_f}  wall=$((t1-t0))s  [$(date '+%F %T')]"
