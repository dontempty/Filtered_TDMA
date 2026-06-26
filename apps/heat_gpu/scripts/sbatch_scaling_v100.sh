#!/bin/bash
#SBATCH --job-name=v100_scale
#SBATCH --partition=cas_v100nv_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --comment inhouse
#SBATCH --output=/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_v100.out
#SBATCH --error=/scratch/x3319a05/Filtered_TDMA/apps/heat_gpu/log/scaling_v100.err

PROJ=/scratch/x3319a05/Filtered_TDMA
BIN="${PROJ}/build/bin/heat_gpu.out"
INP="${PROJ}/apps/heat_gpu/inputs/scaling"
# Output dir = scaling_gpu/<hardware>  (auto-detected; fallback v100 for this script)
GPU_RAW=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)
case "${GPU_RAW}" in
    *V100*) GPU_TAG=v100 ;; *A100*) GPU_TAG=a100 ;;
    *GH200*) GPU_TAG=gh200 ;; *H200*) GPU_TAG=h200 ;; *H100*) GPU_TAG=h100 ;;
    *) GPU_TAG=$(printf '%s' "${GPU_RAW}" | tr -cs 'A-Za-z0-9' '_' | sed 's/^_//;s/_$//') ;;
esac
RES="${PROJ}/apps/heat_gpu/results/scaling_gpu/${GPU_TAG:-v100}"

mkdir -p "${RES}" "${PROJ}/apps/heat_gpu/log"

echo "=== START $(date '+%F %T') ==="
echo "Node: $(hostname)  JOB=${SLURM_JOB_ID}"

module purge
module load nvhpc/25.11_cuda12

export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"

NODE=$(hostname)
echo "GPUs: $(nvidia-smi -L | wc -l)"
nvidia-smi -L
echo "Binary: $(ls -lh ${BIN})"

run_case() {
    local label=$1 np=$2 inp_file=$3
    local tag; tag=$(basename "${inp_file}" .txt)
    echo ""
    echo "---- ${label}  np=${np}  [$(date +%H:%M:%S)] ----"
    TIMING_CSV="${RES}/timing_${tag}.csv" \
        mpirun ${MPI_FLAGS} --host "${NODE}:4" -np ${np} "${BIN}" "${inp_file}" 2>&1
    local rc=$?; [ ${rc} -ne 0 ] && echo "[ERROR] exit code ${rc}"
    return 0
}

echo ""; echo "======== STRONG  np=1,2,4 ========"
for rho in rho025 rho040; do for backend in pascal filtered_v2; do
    echo ""; echo "=== rho=${rho}  backend=${backend} ==="
    for np in 1 2 4; do
        run_case "strong 64²×512"   ${np} "${INP}/strong_64x64x512_np${np}_${backend}_${rho}.txt"
        run_case "strong 128²×1024" ${np} "${INP}/strong_128x128x1024_np${np}_${backend}_${rho}.txt"
        run_case "strong 256²×2048" ${np} "${INP}/strong_256x256x2048_np${np}_${backend}_${rho}.txt"
    done
done; done

echo ""; echo "======== WEAK  np=1,2,4 ========"
for rho in rho025 rho040; do for backend in pascal filtered_v2; do
    echo ""; echo "=== rho=${rho}  backend=${backend} ==="
    for np in 1 2 4; do
        run_case "weak 128³/GPU" ${np} "${INP}/weak_128cube_np${np}_${backend}_${rho}.txt"
        run_case "weak 256³/GPU" ${np} "${INP}/weak_256cube_np${np}_${backend}_${rho}.txt"
        run_case "weak 512³/GPU" ${np} "${INP}/weak_512cube_np${np}_${backend}_${rho}.txt"
    done
done; done

echo ""; echo "======== REFINEMENT  np=2 고정 ========"
for rho in rho025 rho040; do for backend in pascal filtered_v2; do
    echo ""; echo "=== rho=${rho}  backend=${backend} ==="
    for nz in 64 128 256; do
        run_case "refine 64×64×${nz}"   2 "${INP}/refine_64x64x${nz}_np2_${backend}_${rho}.txt"
    done
    for nz in 128 256 512; do
        run_case "refine 128×128×${nz}" 2 "${INP}/refine_128x128x${nz}_np2_${backend}_${rho}.txt"
    done
    for nz in 256 512 1024; do
        run_case "refine 256×256×${nz}" 2 "${INP}/refine_256x256x${nz}_np2_${backend}_${rho}.txt"
    done
done; done

echo ""; echo "==== ALL DONE [$(date '+%F %T')] ===="
echo " results: ${RES}/"
