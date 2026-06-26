#!/bin/bash
#SBATCH -J ftdma_cpu_scaling
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --time=08:00:00
#SBATCH --comment etc
#SBATCH -o /scratch/x3319a05/Filtered_TDMA/apps/heat_cpu/log/scaling_cpu_%j.out
#SBATCH -e /scratch/x3319a05/Filtered_TDMA/apps/heat_cpu/log/scaling_cpu_%j.err

# ============================================================
#  Strong & Weak scaling: Filtered_TDMA (v2) vs PaScaL_TDMA
#  CPU: OpenMPI on cpu partition (48-core nodes, 1 node exclusive)
#  Decomposition: 1D in z (npx=1, npy=1, npz=nproc)
#  rho_z = 0.25 / 0.40 고정 (dt는 코드에서 rho+dz로 자동 계산)
#  Nt=30 total (warmup=10, timed=20)
#
#  Strong (GPU 동일 그리드):
#    64²×512,  128²×1024,  256²×2048    @  np=2,4,8
#  Weak (CPU 맞춤 per-proc):
#    32³/proc, 64³/proc,   128³/proc    @  np=2,4,8
#  Refinement (npz=2 고정):
#    64×64×{64,128,256},  128×128×{128,256,512},  256×256×{256,512,1024}
# ============================================================

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/heat.out"
INP="${PROJ}/apps/heat_cpu/inputs/scaling"
LOG="${PROJ}/apps/heat_cpu/log"
RES="${PROJ}/apps/heat_cpu/results"

mkdir -p "${LOG}" "${RES}"

echo "================================================================"
echo " host       : $(hostname)"
echo " date       : $(date '+%F %T')"
echo " job_id     : ${SLURM_JOB_ID:-interactive}"
echo " partition  : ${SLURM_JOB_PARTITION:-unknown}"
echo " ntasks     : ${SLURM_NTASKS:-?}"
echo "================================================================"

module purge
module load gcc/15.2.0 mpi/openmpi-4.1.8

# ── Binary check ─────────────────────────────────────────────────────────────
if [ ! -f "${BIN}" ]; then
    echo "ERROR: ${BIN} not found. Build with 'make heat' first."
    exit 1
fi
echo "Binary: $(ls -lh ${BIN})"

# ── Runner ────────────────────────────────────────────────────────────────────
run_case() {
    local label=$1 np=$2 inp_file=$3
    local tag
    tag=$(basename "${inp_file}" .txt)
    echo ""
    echo "---- ${label}  np=${np}  [$(date +%H:%M:%S)] ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun -np ${np} "${BIN}" "${inp_file}" 2>&1
    local rc=$?
    [ ${rc} -ne 0 ] && echo "[ERROR] exit code ${rc} — skipping"
    return 0
}

# ── Strong scaling ───────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " STRONG SCALING"
echo "========================================================"

for rho in rho025 rho040; do
for backend in pascal filtered_v2; do
    echo ""
    echo "=== rho=${rho}  backend: ${backend} ==="
    for np in 2 4 8; do
        run_case "strong 64²×512"   ${np} "${INP}/strong_64x64x512_np${np}_${backend}_${rho}.txt"
        run_case "strong 128²×1024" ${np} "${INP}/strong_128x128x1024_np${np}_${backend}_${rho}.txt"
        run_case "strong 256²×2048" ${np} "${INP}/strong_256x256x2048_np${np}_${backend}_${rho}.txt"
    done
done
done

# ── Weak scaling ─────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " WEAK SCALING (32³, 64³, 128³ per proc)"
echo "========================================================"

for rho in rho025 rho040; do
for backend in pascal filtered_v2; do
    echo ""
    echo "=== rho=${rho}  backend: ${backend} ==="
    for np in 2 4 8; do
        run_case "weak 32³/proc"  ${np} "${INP}/weak_32cube_np${np}_${backend}_${rho}.txt"
        run_case "weak 64³/proc"  ${np} "${INP}/weak_64cube_np${np}_${backend}_${rho}.txt"
        run_case "weak 128³/proc" ${np} "${INP}/weak_128cube_np${np}_${backend}_${rho}.txt"
    done
done
done

# ── Refinement ────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " REFINEMENT (npz=2 고정)"
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
echo " log: ${LOG}/scaling_cpu_${SLURM_JOB_ID:-?}.out"
