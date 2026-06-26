#!/bin/bash
#SBATCH -J ftdma_scaling
#SBATCH -p amd_a100nv_8
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --comment etc
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_a100_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_a100_%j.err

# ============================================================
#  Strong & Weak scaling: Filtered_TDMA (v2) vs PaScaL_TDMA
#  GPU: A100 NVLink ×8 (single node, exclusive)
#  Decomposition: 1D in z (npx=1, npy=1, npz=nGPU)
#  rho_z = 0.25 고정 (dt는 코드에서 rho+dz로 자동 계산)
#  Nt=30 total (warmup=10, timed=20)
#
#  Strong scaling (Fig. 6):
#    64²×512,  128²×1024,  256²×2048  @  np=2,4,8
#  Weak scaling (Fig. 7):
#    128³/GPU, 256³/GPU, 512³/GPU     @  np=2,4,8
# ============================================================

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
LOG="${PROJ}/apps/heat_gpu/log"

# Output dir = scaling_gpu/<hardware>  (auto-detected; fallback a100 for this script)
GPU_RAW=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)
case "${GPU_RAW}" in
    *V100*) GPU_TAG=v100 ;; *A100*) GPU_TAG=a100 ;;
    *GH200*) GPU_TAG=gh200 ;; *H200*) GPU_TAG=h200 ;; *H100*) GPU_TAG=h100 ;;
    *) GPU_TAG=$(printf '%s' "${GPU_RAW}" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//') ;;
esac
RES="${PROJ}/apps/heat_gpu/results/scaling_gpu/${GPU_TAG:-a100}"

mkdir -p "${LOG}" "${RES}"

echo "================================================================"
echo " host       : $(hostname)"
echo " date       : $(date '+%F %T')"
echo " job_id     : ${SLURM_JOB_ID:-interactive}"
echo " partition  : ${SLURM_JOB_PARTITION:-unknown}"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -8
echo "================================================================"

module purge
module load nvhpc/25.11_cuda12

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll"

# ── Build (sm_80 = A100) ──────────────────────────────────────────────────────
echo ""
echo "==== [BUILD] heat_gpu (sm_80) ===="
cd "${PROJ}"
make clean >/dev/null 2>&1 || true
USE_CUDA=1 CUDA_ARCH=80 make heat_gpu 2>&1 | tail -5
if [ ! -f "${BIN}" ]; then
    echo "ERROR: build failed, aborting."
    exit 1
fi
ls -lh "${BIN}"

# ── Runner ────────────────────────────────────────────────────────────────────
run_case() {
    local label=$1 np=$2 inp_file=$3
    local tag; tag=$(basename "${inp_file}" .txt)
    echo ""
    echo "---- ${label}  np=${np}  [$(date +%H:%M:%S)] ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} -np ${np} "${BIN}" "${inp_file}" 2>&1
    local rc=$?
    [ ${rc} -ne 0 ] && echo "[ERROR] exit code ${rc} (OOM or crash) — skipping"
    return 0
}

# ── Strong scaling (Fig. 6) ───────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " STRONG SCALING (Fig. 6)"
echo "========================================================"

for rho in rho025 rho040; do
for backend in pascal filtered_v2; do
    echo ""
    echo "=== rho=${rho}  backend: ${backend} ==="
    for np in 2 4 8; do
        run_case "strong 64²×512"    ${np} "${INP}/strong_64x64x512_np${np}_${backend}_${rho}.txt"
        run_case "strong 128²×1024"  ${np} "${INP}/strong_128x128x1024_np${np}_${backend}_${rho}.txt"
        run_case "strong 256²×2048"  ${np} "${INP}/strong_256x256x2048_np${np}_${backend}_${rho}.txt"
    done
done
done

# ── Weak scaling (Fig. 7) ─────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " WEAK SCALING (Fig. 7)"
echo "========================================================"

for rho in rho025 rho040; do
for backend in pascal filtered_v2; do
    echo ""
    echo "=== rho=${rho}  backend: ${backend} ==="
    for np in 2 4 8; do
        run_case "weak 128³/GPU"  ${np} "${INP}/weak_128cube_np${np}_${backend}_${rho}.txt"
        run_case "weak 256³/GPU"  ${np} "${INP}/weak_256cube_np${np}_${backend}_${rho}.txt"
        run_case "weak 512³/GPU"  ${np} "${INP}/weak_512cube_np${np}_${backend}_${rho}.txt"
    done
done
done

# ── Refinement ────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " REFINEMENT (nx,ny 고정 / nz×np 변화 / 1D z 분할)"
echo "========================================================"

for rho in rho025 rho040; do
for backend in pascal filtered_v2; do
    echo ""
    echo "=== rho=${rho}  backend: ${backend} ==="

    for nz in 64 128 256; do
        run_case "refine 64×64×${nz}"   2 "${INP}/refine_64x64x${nz}_np2_${backend}_${rho}.txt"
    done
    for nz in 128 256 512; do
        run_case "refine 128×128×${nz}" 2 "${INP}/refine_128x128x${nz}_np2_${backend}_${rho}.txt"
    done
    for nz in 256 512 1024; do
        run_case "refine 256×256×${nz}" 2 "${INP}/refine_256x256x${nz}_np2_${backend}_${rho}.txt"
    done
done
done

echo ""
echo "==== ALL DONE [$(date '+%F %T')] ===="
echo " results: ${RES}/"
echo " log: ${LOG}/scaling_a100_${SLURM_JOB_ID:-?}.out"
